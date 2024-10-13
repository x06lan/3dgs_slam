import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import sleep
from typing import Union, Tuple, Optional

# from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# from pytorch3d.renderer import PerspectiveCameras

from .cover import CoverSplatter
from .gaussian import Gaussians

from parser.dataset import ColmapDataset
from utils.image import ImageInfo
from utils.camera import Camera
from utils.point import Point3D
from utils.function import save_image, resize_image, normalize, maxmin_normalize


class Trainer:
    def __init__(self, ckpt: Union[str, None] = None, lr: float = 0.003, downsample: int = 4, distance: int = 8, camera: Camera = None):
        self.lr = lr

        self.downsample = downsample

        self.splatter = CoverSplatter(
            load_ckpt=ckpt, downsample=self.downsample, grid_downsample=distance)

        if camera is not None:
            self.splatter.set_camera(camera)

        self.ssim = StructuralSimilarityIndexMeasure(
            reduction="elementwise_mean").to(self.splatter.device)

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.depth_cache = {}

    def depth_normalize(self, depth: torch.Tensor):
        depth = depth.to(torch.float)
        depth, _ = normalize(depth)
        depth = maxmin_normalize(depth)
        return depth

    def critic(self, source, target):
        ssim_weight = 0.7

        # map B,H,W,C to B,C,H,W
        ssim = self.ssim(source.unsqueeze(0).permute(
            0, 3, 1, 2), target.unsqueeze(0).permute(0, 3, 1, 2))
        ssim_loss = 1 - ssim

        l2_loss = self.l2(source, target)

        return ssim_weight * ssim_loss + (1 - ssim_weight) * l2_loss
        # return self.l2(source, target)

    def step(self, image_info: ImageInfo, ground_truth: torch.Tensor, cover: bool = False, grad: bool = True) -> Tuple[torch.Tensor, dict]:

        # assert ground_truth.device == self.splatter.device
        params = [
            {"params": self.splatter.gaussians.pos, "lr": self.lr*2.0},
            {"params": self.splatter.gaussians.rgb, "lr": self.lr},
            {"params": self.splatter.gaussians.scale, "lr": self.lr*1.5},
            {"params": self.splatter.gaussians.quaternion, "lr": self.lr},
            {"params": self.splatter.gaussians.opacity, "lr": self.lr},
        ]

        self.optimizer = torch.optim.Adam(
            params=params, betas=(0.9, 0.99))

        if not grad:
            with torch.no_grad():
                render_image, gt_depth = self.splatter(
                    image_info, ground_truth, cover)
                return render_image, {}
        else:
            render_image, gt_depth = self.splatter(
                image_info, ground_truth, cover)

            if gt_depth is not None and image_info.id not in self.depth_cache:

                # resize gt_depth to match render_image
                gt_depth = resize_image(
                    gt_depth, render_image.shape[1], render_image.shape[0])

                self.depth_cache[image_info.id] = gt_depth

        render_image_rgb = render_image[..., :3]
        ground_truth = ground_truth.to(self.splatter.device)

        render_depth = render_image[..., 4]
        render_depth = self.depth_normalize(render_depth)
        render_depth = render_depth.unsqueeze(-1)

        rgb_loss = self.critic(render_image_rgb, ground_truth)
        try:
            gt_depth = self.depth_cache[image_info.id].to(self.splatter.device)
            gt_depth = self.depth_normalize(gt_depth)

            depth_loss = self.critic(render_depth, gt_depth)
            loss = rgb_loss + depth_loss
            # loss = rgb_loss
        except:
            loss = rgb_loss

        dump = {
            "loss": loss.item(),
            "count": self.splatter.gaussians.pos.shape[0],
            "lr": self.optimizer.param_groups[0]["lr"],
            # "depth_paramter": self.splatter.depth_paramter.tolist(),
            # "depth": render_depth[0],
        }

        if not torch.isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return render_image, dump

    def eval(self):
        pass


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

        return self.data[0: length: self.stride]


if __name__ == "__main__":
    frame = 0
    downsample = 2
    lr = 0.002
    # lr = 0.0005
    batch = 30
    grid_downsample = 8
    refine = False
    # refine = True
    test = False
    # test = True

    if test:
        batch = 1

    # dataset = ColmapDataset("./record",
    dataset = ColmapDataset(
        "dataset/nerfstudio/poster",
        # "dataset/nerfstudio/record",
        # "dataset/nerfstudio/stump",
        # "dataset/nerfstudio/aspen",
        # "dataset/nerfstudio/redwoods2",
        # "dataset/nerfstudio/person",
        # "dataset/nerfstudio/library",
        # "dataset/nerfstudio/plane",
        # "dataset/nerfstudio/Egypt",
        # "dataset/nerfstudio/storefront",
        downsample_factor=downsample,
    )

    if test:
        trainer = Trainer(
            ckpt="3dgs_slam_ckpt.pth",
            # trainer = Trainer(ckpt="3dgs_slam_ckpt_refine.pth",
            camera=dataset.camera,
            lr=lr,
            downsample=downsample,
        )
    else:
        trainer = Trainer(camera=dataset.camera, lr=lr,
                          downsample=downsample, distance=grid_downsample)

    bar = tqdm(range(0, len(dataset.image_info)))
    dm = DataManager(batch=batch, stride=5)

    window_list = []
    for img_id in bar:
        frame = img_id

        ground_truth = dataset.images[frame]
        ground_truth = ground_truth.to(torch.float).to(
            trainer.splatter.device) / 255

        image_info = dataset.image_info[frame]
        # print(image_info.id)
        dm.add_image(ground_truth, image_info)

        status = {}

        # for info, gt in reversed(window_list):
        for it, (gt, info) in enumerate(dm.get_train_data()):
            # print(info.id)

            retrain_count = 1
            if test:
                retrain_count = 1

            for i in range(retrain_count):
                if test:
                    grad = False
                    cover = False
                else:
                    grad = True
                    cover = it == 0 and i == 0

                render_image, status = trainer.step(
                    image_info=info, ground_truth=gt, cover=cover, grad=grad)
            if status:
                status["count"] = f'{status["count"]/1000 :.2f}k'
        bar.set_postfix(status)
        # save image
        save_image("output.png", render_image[..., :3])

        if img_id % 1 == 0 and not test:
            control_status = trainer.splatter.adaptive_control()
            # status.update(control_status)
            print(control_status)
        # bar.set_postfix(status)

        depth_image = render_image[..., 4].detach().unsqueeze(-1)
        depth_image = maxmin_normalize(depth_image)
        # turn depth to rgb with jet colormap
        depth_image = depth_image.squeeze().cpu().numpy()
        # turn dpeht to rgb with jet colormap
        depth_image = cv2.applyColorMap(
            (depth_image * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # depth_image = depth_image.repeat(1, 1, 3)
        save_image("depth_output.png", depth_image)

        if not test:
            trainer.splatter.save_ckpt("3dgs_slam_ckpt.pth")
        else:
            sleep(0.05)
