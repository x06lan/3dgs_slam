import cv2
import time
import torch
import psutil
import os
import shutil
import numpy as np
from tqdm import tqdm
import threading
import multiprocessing
import multiprocessing.shared_memory

import enlighten
from pathlib import Path
import pycolmap

from utils.camera import Camera
from utils.function import resize_image, save_image, euler_to_quaternion, convert_z_up_to_y_up
from utils.image import ImageInfo
from parser.dataset import ColmapDataset
from viewer.server import Viewer, ViewerData

from splatting.trainer import Trainer

# TODO: run dataset (?)
# TODO: send stage to viewer

SKIP_COLMAP = True


def close_port(port):
    from psutil import process_iter
    from signal import SIGKILL  # or SIGKILL

    for proc in process_iter():
        for conns in proc.net_connections(kind="inet"):
            if conns.laddr.port == port:
                print("Process ID: ", proc.pid)
                proc.send_signal(SIGKILL)  # or SIGKILL


class DataManager:
    def __init__(self, batch=40, stride=5):
        self.data = []
        self.batch = batch
        self.stride = stride
        self.max_length = 200

    def add_image(self, image, image_info):
        self.data.insert(0, (image, image_info))
        if len(self.data) >= self.max_length:
            self.data.pop()

    def get_train_data(self):
        length = len(self.data)
        length = min(length, self.batch * self.stride)

        return self.data[0 : length : self.stride]


class Colmap:
    def __init__(self, data, data_dir="record"):
        self.shareData: ViewerData = data
        self.dir = data_dir

    def run(self):
        while True:
            self.shareData.require()
            if self.shareData.stage == 2:
                self.shareData.release()
                print("COLMAP START")
                output_path = Path(self.dir) / "colmap"
                image_path = Path(self.dir) / "images"
                sparse_path = output_path / "sparse"
                database_path = output_path / "database.db"

                output_path.mkdir(exist_ok=True)
                pycolmap.extract_features(database_path, image_path)
                pycolmap.match_exhaustive(database_path)
                num_images = pycolmap.Database(database_path).num_images
                with enlighten.Manager() as manager:
                    with manager.counter(total=num_images, desc="Images registered:") as pbar:
                        pbar.update(0, force=True)
                        recs = pycolmap.incremental_mapping(
                            database_path,
                            image_path,
                            sparse_path,
                            initial_image_pair_callback=lambda: pbar.update(2),
                            next_image_callback=lambda: pbar.update(1),
                        )
                print("COLMAP FINISH")

                # end colmap, to train
                self.shareData.require()
                self.shareData.stage = 3
                self.shareData.release()
                break
            self.shareData.release()
            time.sleep(0.05)


class Tracker:
    def __init__(self, data, data_dir="record"):
        self.shareData: ViewerData = data
        self.preview = True
        self.batch = 10
        self.lr = 0.005
        self.stride = 5

        self.data_dir = data_dir
        self.run_dataset = False
        self.dataset_dir = self.data_dir
        # self.dataset_dir = "dataset/nerfstudio/poster"
        # self.dataset_dir = "dataset/nerfstudio/stump"
        # self.dataset_dir = "dataset/nerfstudio/aspen"
        # self.dataset_dir = "dataset/nerfstudio/redwoods2"
        self.ckpt = "test_3dgs_slam_ckpt.pth"

        self.reset()

    def reset(self):
        self.camera = None
        self.datamanager = DataManager(batch=self.batch, stride=self.stride)

        self.is_loaded_dataset = False
        self.train_progress = 1
        self.record_count = 0

        if not SKIP_COLMAP:
            self.clear_data_dir()

    def clear_data_dir(self):
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir)
        os.makedirs(self.data_dir + "/images")

    def load_dataset(self, preview, grid, downsample=4):
        self.downsample = downsample
        self.grid = grid
        self.downscale_dataset(downsample)
        if preview:
            self.dataset = ColmapDataset(self.dataset_dir, downsample_factor=downsample)
            self.camera = self.dataset.camera
            self.trainer = Trainer(ckpt=self.ckpt, camera=self.camera, lr=self.lr, downsample=self.downsample, distance=self.grid)
        else:
            # print(width, height)
            self.dataset = ColmapDataset(self.dataset_dir, downsample_factor=downsample)
            self.camera = self.dataset.camera
            self.trainer = Trainer(camera=self.camera, lr=self.lr, downsample=downsample)

    def downscale_dataset(self, downscale):
        oringin_dir = f"{self.dataset_dir}/images"
        new_dir = f"{self.dataset_dir}/images_{downscale}"
        if os.path.exists(new_dir) or downscale == 1:
            return
        os.mkdir(new_dir)
        for filename in tqdm(os.listdir(oringin_dir)):
            img = cv2.imread(os.path.join(oringin_dir, filename))
            img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
            cv2.imwrite(os.path.join(new_dir, filename), img)

    def run(self):
        while True:
            self.shareData.require()

            # IDLE
            if self.shareData.stage == 0:
                # start record
                if self.shareData.play:
                    self.shareData.stage = 1
                if SKIP_COLMAP:
                    self.shareData.stage = 3

            # RECORD
            elif self.shareData.stage == 1:
                if self.shareData.play == False:
                    # stop record, start colmap
                    self.shareData.stage = 2
                    self.shareData.release()
                    continue

                if self.shareData.image_update:
                    image = self.shareData.recive_image
                    save_image(f"{self.data_dir}/images/{self.record_count}.png", image)
                    print(f"record {self.record_count}: {image.shape}")
                    self.record_count += 1

                    self.shareData.render_height = image.shape[0]
                    self.shareData.render_width = image.shape[1]
                    share_image = self.shareData.render_image
                    share_image[:] = image[:]

                    self.shareData.image_update = False

            # COLMAP
            elif self.shareData.stage == 2:
                self.shareData.release()
                continue

            # TRAIN
            elif self.shareData.stage == 3:
                if not self.is_loaded_dataset:
                    preview = False
                    grid = self.shareData.grid
                    downsample = self.shareData.downsample
                    self.load_dataset(preview, grid, downsample)
                    self.is_loaded_dataset = True

                if self.train_progress == len(self.dataset.images):
                    # finish train, to preview
                    self.trainer.splatter.save_ckpt(self.ckpt)
                    self.shareData.stage = 4
                    self.shareData.release()
                    continue

                display_image = None
                ground_truth = self.dataset.images[self.train_progress].to(torch.float).to(self.trainer.splatter.device) / 255
                image_info = self.dataset.image_info[self.train_progress]
                print(image_info)
                self.datamanager.add_image(ground_truth, image_info)

                for i, (gt, info) in enumerate(self.datamanager.get_train_data()):
                    cover = i == 0
                    grad = True
                    render_image, status = self.trainer.step(image_info=info, ground_truth=gt, cover=cover, grad=grad)
                    # print(i, info.id, status)
                    if self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]:
                        self.shareData.render_width = render_image.shape[1]
                        self.shareData.render_height = render_image.shape[0]
                    display_image = render_image[..., :3]
                    print(status)
                self.train_progress += 1

                print(display_image.shape)
                display_image = (display_image).detach().cpu().numpy()
                display_image = (display_image * 255).astype(np.uint8)
                self.shareData.render_width = display_image.shape[1]
                self.shareData.render_height = display_image.shape[0]
                share_image = self.shareData.render_image
                share_image[:] = display_image[:]

            # PREVIEW
            elif self.shareData.stage == 4:
                # ground_truth = torch.from_numpy(
                # self.shareData.recive_image)
                qvec = euler_to_quaternion(self.shareData.rotation[2] + 180, self.shareData.rotation[1] + 180, self.shareData.rotation[0] + 180)
                # rotate 90 degree
                # qvec = convert_z_up_to_y_up(qvec)
                tvec = torch.tensor(list(self.shareData.position)).to(torch.float).to(self.trainer.splatter.device)
                # print(tvec)
                # tvec = torch.zeros(3).to(self.trainer.splatter.device)

                ground_truth = None
                image_info = self.dataset.image_info[1]
                image_info.qvec = torch.from_numpy(qvec)
                image_info.tvec = tvec
                cover = False
                grad = False
                render_image, status = self.trainer.step(image_info=image_info, ground_truth=ground_truth, cover=cover, grad=grad)
                if self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]:
                    self.shareData.render_width = render_image.shape[1]
                    self.shareData.render_height = render_image.shape[0]
                display_image = render_image[..., :3]
                save_image("output.png", display_image)

                display_image = (display_image).detach().cpu().numpy()
                display_image = (display_image * 255).astype(np.uint8)
                share_image = self.shareData.render_image
                share_image[:] = display_image[:]

            self.shareData.image_update = False
            self.shareData.release()
            time.sleep(0.05)


if __name__ == "__main__":

    close_port(8000)

    data = ViewerData()
    viewer = Viewer(data=data)
    tracker = Tracker(data)
    colmap = Colmap(data)

    viewer_thread = multiprocessing.Process(target=viewer.run, args=("0.0.0.0", 8000))
    tracker_thread = multiprocessing.Process(target=tracker.run, args=())

    if not SKIP_COLMAP:
        colmap_thread = multiprocessing.Process(target=colmap.run, args=())

    viewer_thread.start()
    tracker_thread.start()

    if not SKIP_COLMAP:
        colmap_thread.start()

    viewer_thread.join()
    tracker_thread.join()

    if not SKIP_COLMAP:
        colmap_thread.join()
