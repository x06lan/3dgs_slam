import multiprocessing.shared_memory
import torch
import cv2
import time
import numpy as np
import threading
import multiprocessing
import os
import shutil
from PIL import Image

import enlighten
from pathlib import Path
import pycolmap

# import gaussian_cuda

from utils.camera import Camera
from parser.dataset import ColmapDataset
from viewer.server import Viewer, ViewerData


def log_info(data):
    while True:
        data.require()

        if data.change:
            print(data.width, data.height, data.change)
            data.change = False
        data.release()

        time.sleep(1)


class Tracker:
    def __init__(self, data):
        self.shareData: ViewerData = data
        # self.camera = None
        self.img_id = 0
        self.is_recording = True

    def run(self):
        if os.path.exists("./record_images"):
            shutil.rmtree("./record_images")
        os.mkdir("./record_images")

        while True:
            if self.is_recording:
                self.shareData.require()
                if self.shareData.image_update:
                    image = self.shareData.image
                    self.shareData.image_update = False

                    image_file = Image.fromarray(image)
                    image_file.save(f"./record_images/{self.img_id}.jpg")
                    self.img_id += 1

                self.shareData.release()
            else:
                run_colmap("./record_images", "./colmap_output")
                break
            time.sleep(0.1)


def run_colmap(image_dir, output_dir):
    output_path = Path(output_dir)
    image_path = Path(image_dir)
    sfm_path = output_path / "sfm"
    database_path = output_path / "database.db"

    output_path.mkdir(exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)
    num_images = pycolmap.Database(database_path).num_images

    pycolmap.incremental_mapping(database_path, image_path, sfm_path)
    # with enlighten.Manager() as manager:
    #     with manager.counter(total=num_images, desc="Images registered:") as pbar:
    #         pbar.update(0, force=True)
    #         recs = pycolmap.incremental_mapping(
    #             database_path,
    #             image_path,
    #             sfm_path,
    #             initial_image_pair_callback=lambda: pbar.update(2),
    #             next_image_callback=lambda: pbar.update(1),
    #         )


if __name__ == "__main__":
    run_colmap("./record_images", "./colmap_output1")
    # data = ViewerData()
    # viewer = Viewer(data=data)
    # tracker = Tracker(data)

    # viewer_thread = multiprocessing.Process(target=viewer.run, args=("0.0.0.0", 6969))
    # # log_thread = multiprocessing.Process(
    # #     target=log_info, args=(data,))
    # tracker_thread = multiprocessing.Process(target=tracker.run, args=())

    # viewer_thread.start()
    # tracker_thread.start()
    # # log_thread.start()

    # viewer_thread.join()
    # tracker_thread.join()
    # # log_thread.join()
