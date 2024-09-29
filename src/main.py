import cv2
import time
import torch
import psutil
import numpy as np
import threading
import multiprocessing
import os
import shutil
from PIL import Image

import enlighten
from pathlib import Path
import pycolmap
import multiprocessing.shared_memory


from utils.camera import Camera
from utils.function import resize_image, save_image, euler_to_quaternion, convert_z_up_to_y_up, downsample_imags
from utils.image import ImageInfo
from parser.dataset import ColmapDataset
from viewer.server import Viewer, ViewerData

from splatting.trainer import Trainer


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


class Tracker:
    def __init__(self, data):

        self.img_id = 1

        # colmap/record args
        self.record_count = 0
        self.record_dir = "./record"
        self.clean_record = True

        # datamanage args
        self.datamanager_batch = 10
        self.datamanager_stride = 5

        # trainer args
        self.trainer_lr = 0.0025

        # dataset args
        self.dataset_dir = self.record_dir
        # self.dataset_dir = "dataset/nerfstudio/poster"
        # self.dataset_dir = "dataset/nerfstudio/stump"
        # self.dataset_dir = "dataset/nerfstudio/aspen"
        # self.dataset_dir = "dataset/nerfstudio/redwoods2"
        self.dataset_downsample = 4
        self.dataset_distance = 8
        self.dataset_ckpt = None  # "3dgs_slam_ckpt.pth"

        self.state = "train"  # record, colmap, train, test

        self.shareData: ViewerData = data
        self.datamanager = DataManager(batch=self.datamanager_batch, stride=self.datamanager_stride)
        self.dataset = None
        self.trainer = None

    def clear_record_dir(self):
        if os.path.exists(self.record_dir):
            shutil.rmtree(self.record_dir)
        os.makedirs(self.record_dir)
        os.makedirs(self.record_dir + "/images")

    def load_dataset(self):
        self.dataset = ColmapDataset(self.dataset_dir, downsample_factor=self.dataset_downsample)

    def setup_trainer(self, setting="train"):
        assert self.dataset is not None
        if setting == "train":
            self.trainer = Trainer(camera=self.dataset.camera, lr=self.trainer_lr, downsample=self.dataset_downsample)
        elif setting == "test":
            self.trainer = Trainer(ckpt=self.dataset_ckpt, camera=self.dataset.camera, lr=self.trainer_lr, downsample=self.dataset_downsample)

    def run(self):
        init = False
        while True:
            if not init:
                self.load_dataset()
                self.setup_trainer()
                # if self.clear_record_dir:
                # self.clear_record_dir()
                init = True

            if self.shareData.recive_width <= 0 or self.shareData.recive_height <= 0:
                time.sleep(0.01)
                continue

            if self.state == "record":
                # self.shareData.require()
                if self.shareData.image_update:
                    self.shareData.image_update = False
                    recive_image = self.shareData.recive_image
                    self.record(recive_image)
                # self.shareData.release()

            elif self.state == "colmap":
                self.run_colmap()

            elif self.state == "train" or self.state == "test":
                self.shareData.require()
                display_image = self.run_gaussian(self.state)
                save_image("output.jpg", display_image)
                display_image = (display_image).detach().cpu().numpy()
                display_image = (display_image * 255).astype(np.uint8)
                share_image = self.shareData.render_image
                share_image[:] = display_image[:]
                self.shareData.release()
            time.sleep(0.01)

    def run_colmap(self):
        output_path = Path(self.record_dir) / "colmap"
        image_path = Path(self.record_dir) / "images"
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
        print("colmap done")

    def record(self, image):
        print(f"{self.record_count} recive {image.shape}")
        # save_image("output.png", image)
        save_image(f"{self.record_dir}/images/{self.record_count}.png", image)
        self.record_count += 1

    def run_gaussian(self, setting="train"):
        display_image = None

        if setting == "test":
            ground_truth = torch.from_numpy(self.shareData.recive_image)
            qvec = euler_to_quaternion(self.shareData.rotation[0], self.shareData.rotation[2], self.shareData.rotation[1])
            qvec = convert_z_up_to_y_up(qvec)

            image_info = self.dataset.image_info[self.img_id]
            image_info.qvec = torch.from_numpy(qvec)
            cover = False
            grad = False
            render_image, status = self.trainer.step(image_info=image_info, ground_truth=ground_truth, cover=cover, grad=grad)
            if self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]:
                self.shareData.render_width = render_image.shape[1]
                self.shareData.render_height = render_image.shape[0]
            display_image = render_image[..., :3]

        else:
            id = self.img_id % (128) + 1

            ground_truth = self.dataset.images[id]
            ground_truth = ground_truth.to(torch.float).to(self.trainer.splatter.device) / 255

            image_info = self.dataset.image_info[id]
            self.datamanager.add_image(ground_truth, image_info)
            # print(image_info)

            for i, (gt, info) in enumerate(self.datamanager.get_train_data()):
                cover = i == 0
                grad = True
                render_image, status = self.trainer.step(image_info=info, ground_truth=gt, cover=cover, grad=grad)
                # print(i, info.id, status)
                if self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]:
                    self.shareData.render_width = render_image.shape[1]
                    self.shareData.render_height = render_image.shape[0]
                display_image = render_image[..., :3]
            self.img_id += 1
            # self.trainer.splatter.save_ckpt("3dgs_slam_ckpt.pth")
        return display_image


def test():
    # downsample_imags("./record/images", 4)
    data = ViewerData()
    tracker = Tracker(data)
    tracker.run_colmap()
    # tracker.load_dataset()
    # tracker.setup_trainer()
    # while True:
    #     image = tracker.run_gaussian()
    #     save_image("output.jpg", image)
    pass


if __name__ == "__main__":
    test()

    # data = ViewerData()
    # viewer = Viewer(data=data)
    # tracker = Tracker(data)

    # close_port(8000)
    # viewer_thread = multiprocessing.Process(target=viewer.run, args=("0.0.0.0", 8000))
    # tracker_thread = multiprocessing.Process(target=tracker.run, args=())
    # # log_thread = multiprocessing.Process(
    # #     target=log_info, args=(data,))

    # viewer_thread.start()
    # tracker_thread.start()
    # # log_thread.start()

    # viewer_thread.join()
    # tracker_thread.join()
    # log_thread.join()
