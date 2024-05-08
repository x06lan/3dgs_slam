import multiprocessing.shared_memory
import torch
import cv2
import time
import numpy as np
import threading
import multiprocessing

# import gaussian_cuda

from utils.camera import Camera
from parser.dataset import ColmapDataset
from tracker.visual_odometry import VisualOdometry
from tracker.feature_tracker_configs import FeatureTrackerConfigs
from tracker.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from viewer.mplot_thread import Mplot2d, Mplot3d
from viewer.server import Viewer, ViewerData


def log_info(data):
    while True:
        data.require()

        if (data.change):
            print(data.width, data.height, data.change)
            data.change = False

        data.release()

        time.sleep(1)


class Tracker():
    def __init__(self, data):
        self.shareData: ViewerData = data
        self.camera = None
        self.vo = None
        self.img_id = 0

    def create_vo(self):
        num_features = 1000  # how many features do you want to detect and track?
        tracker_config = FeatureTrackerConfigs.LK_FAST
        tracker_config["num_features"] = num_features
        feature_tracker = feature_tracker_factory(**tracker_config)
        width = self.shareData.width
        height = self.shareData.height
        self.camera = Camera(width=width, height=height, cx=width/2,
                             cy=height/2, fx=width/2, fy=height/2, distortParams=[0, 0, 0, 0, 0], fps=30)
        self.vo = VisualOdometry(self.camera, feature_tracker, None)

    def run(self):

        while True:
            if (self.vo is None and self.shareData.width > 0 and self.shareData.height > 0):
                self.create_vo()

            if (self.vo is not None):

                self.shareData.require()
                # print(self.data.width, self.data.height, self.data.image_update)

                if (self.shareData.image_update):
                    image = self.shareData.image
                    self.shareData.image_update = False
                    self.vo.track(image, self.img_id)

                    if (len(self.vo.traj3d_est) > 0):
                        t = self.vo.traj3d_est[-1].T[0]
                        self.shareData.position[0] = float(t[0])
                        self.shareData.position[1] = float(t[1])
                        self.shareData.position[2] = float(t[2])
                        # print(t)

                self.shareData.release()
                self.img_id += 1
            time.sleep(0.1)


if __name__ == "__main__":

    vo = None

    data = ViewerData()
    viewer = Viewer(data=data)
    tracker = Tracker(data)

    viewer_thread = multiprocessing.Process(
        target=viewer.run, args=("0.0.0.0", 8000))
    # log_thread = multiprocessing.Process(
    #     target=log_info, args=(data,))
    tracker_thread = multiprocessing.Process(
        target=tracker.run, args=())

    viewer_thread.start()
    tracker_thread.start()
    # log_thread.start()

    viewer_thread.join()
    tracker_thread.join()
    # log_thread.join()
