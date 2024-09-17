import cv2
import time
import torch
import psutil
import numpy as np
import threading
import multiprocessing
import multiprocessing.shared_memory


from utils.camera import Camera
from utils.function import resize_image, save_image, euler_to_quaternion
# from utils.image import ImageInfo
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


class Tracker():
    def __init__(self, data):
        self.shareData: ViewerData = data
        self.inited = False
        self.camera = None
        self.img_id = 1

        self.test = True
        self.batch = 40
        self.lr = 0.0005

    def init(self):
        self.inited = True
        width = self.shareData.recive_width
        height = self.shareData.recive_height
        self.downsample = 2
        self.dataset = ColmapDataset("dataset/nerfstudio/poster",
                                     # self.dataset=ColmapDataset("dataset/nerfstudio/stump",
                                     # dataset = ColmapDataset("dataset/nerfstudio/aspen",
                                     # dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
                                     # dataset = ColmapDataset("dataset/nerfstudio/person",
                                     downsample_factor=self.downsample)
        self.camera = self.dataset.camera
        # self.camera = Camera(width=width, height=height, cx=width/2,
        #                      cy=height/2, fx=width/2, fy=height/2, distortParams=[0, 0, 0, 0, 0], fps=30)
        if self.test:
            self.trainer = Trainer(ckpt="3dgs_slam_ckpt.pth",
                                   camera=self.camera, lr=self.lr, downsample=self.downsample)
        else:
            self.trainer = Trainer(
                camera=self.camera, lr=self.lr, downsample=self.downsample)

    def run(self):

        while True:
            if (not self.inited and self.shareData.recive_width > 0 and self.shareData.recive_height > 0):
                self.init()

            if (self.inited):

                self.shareData.require()
                # print(self.data.width, self.data.height, self.data.image_update)

                if (self.shareData.image_update):
                    self.shareData.image_update = False
                    cover = False
                    grad = False
                    id = self.img_id % (200)+1
                    # print("image_update", id)
                    ground_truth = torch.from_numpy(
                        self.shareData.recive_image)

                    ground_truth = self.dataset.images[id]
                    ground_truth = ground_truth.to(torch.float).to(
                        self.trainer.splatter.device)/255

                    image_info = self.dataset.image_info[id]
                    # image_info.tvec[0] = self.shareData.position[0]
                    # image_info.tvec[1] = self.shareData.position[1]
                    # image_info.tvec[2] = self.shareData.position[2]

                    image_info.tvec[0] = 0
                    image_info.tvec[1] = 0
                    image_info.tvec[2] = 0

                    qvec = euler_to_quaternion(
                        self.shareData.rotation[0], self.shareData.rotation[2], self.shareData.rotation[1])

                    def convert_z_up_to_y_up(quaternion):
                        w = quaternion[0]
                        x = quaternion[1]
                        y = quaternion[2]
                        z = quaternion[3]

                        quaternion[1] = z
                        quaternion[2] = -y
                        quaternion[3] = x
                        return quaternion
                    qvec = convert_z_up_to_y_up(qvec)
                    # qvec = np.array(self.shareData.rotation)
                    image_info.qvec = torch.from_numpy(qvec)

                    print("image_info", image_info.id, qvec)

                    render_image, _ = self.trainer.step(image_info=image_info,
                                                        ground_truth=ground_truth, cover=cover, grad=grad)

                    render_image = render_image[..., :3]
                    # render_image = resize_image(
                    #     render_image, self.shareData.recive_width, self.shareData.recive_height)

                    # save_image("render_image.png", render_image)

                    if (self.shareData.render_width != render_image.shape[1] or self.shareData.render_height != render_image.shape[0]):

                        self.shareData.render_width = render_image.shape[1]
                        self.shareData.render_height = render_image.shape[0]

                    share_image = self.shareData.render_image

                    render_image = render_image.cpu().numpy()*255
                    render_image = render_image.astype(np.uint8)
                    share_image[:] = render_image[:]
                    # print("render", self.shareData.render_image[0][:5])

                self.shareData.release()
                self.img_id += 1
            # time.sleep(0.1)


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
