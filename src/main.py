import cv2
import time
import torch
import psutil
import numpy as np
import threading
import multiprocessing
import multiprocessing.shared_memory


from utils.camera import Camera
from utils.function import resize_image, save_image, euler_to_quaternion, convert_z_up_to_y_up
from utils.image import ImageInfo
from parser.dataset import ColmapDataset
from viewer.server import Viewer, ViewerData

from splatting.trainer import Trainer


def close_port(port):
    from psutil import process_iter
    from signal import SIGKILL  # or SIGKILL

    for proc in process_iter():
        for conns in proc.net_connections(kind='inet'):
            if conns.laddr.port == port:
                print('Process ID: ', proc.pid)
                proc.send_signal(SIGKILL)  # or SIGKILL


class DataManager():
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
        length = min(length, self.batch*self.stride)

        return self.data[0:length:self.stride]


class Tracker():
    def __init__(self, data):
        self.shareData: ViewerData = data
        self.inited = False
        self.camera = None
        self.img_id = 1

        self.test = False
        self.batch = 10
        self.lr = 0.0025
        self.stride = 5

        self.datamanager = DataManager(batch=self.batch, stride=self.stride)

    def init(self):
        self.inited = True
        self.downsample = 2
        self.distance = 8
        if self.test:
            self.dataset = ColmapDataset("dataset/nerfstudio/poster",
                                         # self.dataset=ColmapDataset("dataset/nerfstudio/stump",
                                         # dataset = ColmapDataset("dataset/nerfstudio/aspen",
                                         # dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
                                         # dataset = ColmapDataset("dataset/nerfstudio/person",
                                         downsample_factor=self.downsample)
            self.camera = self.dataset.camera
            self.trainer = Trainer(ckpt="3dgs_slam_ckpt.pth",
                                   camera=self.camera, lr=self.lr, downsample=self.downsample, distance=self.distance)
        else:
            # width = self.shareData.recive_width
            # height = self.shareData.recive_height
            # self.camera = Camera(width=width, height=height, cx=width/2,
            #                      cy=height/2, fx=width/2, fy=height/2, distortParams=[0, 0, 0, 0, 0], fps=30)
            # self.dataset = ColmapDataset("dataset/nerfstudio/poster",
            # self.dataset = ColmapDataset("dataset/nerfstudio/stump",
            # self.dataset = ColmapDataset("dataset/nerfstudio/aspen",
            # self.dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
            self.dataset = ColmapDataset("dataset/nerfstudio/person",
                                         downsample_factor=self.downsample)
            self.camera = self.dataset.camera
            self.trainer = Trainer(
                camera=self.camera, lr=self.lr, downsample=self.downsample)

    def run(self):

        while True:
            # with self.shareData:

            if (not self.inited and self.shareData.recive_width > 0 and self.shareData.recive_height > 0):
                self.init()
            elif not self.inited:
                continue

            # self.shareData.require()
            if (self.shareData.image_update):
                self.shareData.image_update = False
                display_image = None

                if self.test:
                    ground_truth = torch.from_numpy(
                        self.shareData.recive_image)
                    qvec = euler_to_quaternion(
                        self.shareData.rotation[0], self.shareData.rotation[2], self.shareData.rotation[1])
                    qvec = convert_z_up_to_y_up(qvec)

                    image_info = self.dataset.image_info[self.img_id]
                    image_info.qvec = torch.from_numpy(qvec)
                    cover = False
                    grad = False
                    render_image, status = self.trainer.step(image_info=image_info,
                                                             ground_truth=ground_truth, cover=cover, grad=grad)
                    if (self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]):
                        self.shareData.render_width = render_image.shape[1]
                        self.shareData.render_height = render_image.shape[0]
                    display_image = render_image[..., :3]

                else:
                    id = self.img_id % (200)+1

                    ground_truth = self.dataset.images[id]
                    ground_truth = ground_truth.to(torch.float).to(
                        self.trainer.splatter.device)/255

                    image_info = self.dataset.image_info[id]
                    self.datamanager.add_image(ground_truth, image_info)

                    for i, (gt, info) in enumerate(self.datamanager.get_train_data()):
                        cover = (i == 0)
                        grad = True
                        render_image, status = self.trainer.step(image_info=info,
                                                                 ground_truth=gt, cover=cover, grad=grad)
                        # print(i, info.id, status)
                        if (self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]):
                            self.shareData.render_width = render_image.shape[1]
                            self.shareData.render_height = render_image.shape[0]
                        display_image = render_image[..., :3]
                    self.img_id += 1
                recive_image = self.shareData.recive_image
                print("recive", recive_image.shape)
                save_image("output.png", recive_image)

                # self.trainer.splatter.save("3dgs_slam_ckpt.pth")
                # print("render", display_image.shape)
                display_image = (display_image).detach().cpu().numpy()
                display_image = (display_image*255).astype(np.uint8)
                share_image = self.shareData.render_image
                share_image[:] = display_image[:]
                # print("share", share_image.shape)
            # self.shareData.release()
            # time.sleep(0.01)


if __name__ == "__main__":

    close_port(8000)

    data = ViewerData()
    viewer = Viewer(data=data)
    tracker = Tracker(data)

    viewer_thread = multiprocessing.Process(
        target=viewer.run, args=("0.0.0.0", 8000))
    tracker_thread = multiprocessing.Process(
        target=tracker.run, args=())

    viewer_thread.start()
    tracker_thread.start()
    # log_thread.start()

    viewer_thread.join()
    tracker_thread.join()
    # log_thread.join()
