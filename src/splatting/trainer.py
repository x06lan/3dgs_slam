
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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


class Trainer():
    def __init__(self, ckpt: Union[str, None] = None, lr: float = 0.003, downsample: int = 4, camera: Camera = None):
        self.lr = lr

        self.downsample = downsample

        self.splatter = CoverSplatter(
            load_ckpt=ckpt, downsample=self.downsample)

        if camera is not None:
            self.splatter.set_camera(camera)

        self.ssim = StructuralSimilarityIndexMeasure(
            reduction="elementwise_mean").to(self.splatter.device)

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.depth_cache = {}

    def depth_normalize(self, depth: torch.Tensor):
        depth = depth.to(torch.float)
        # depth, _ = normalize(depth)
        depth = maxmin_normalize(depth)
        return depth

    def critic(self, source, target):
        ssim_weight = 0.5

        # map B,H,W,C to B,C,H,W
        ssim = self.ssim(source.unsqueeze(0).permute(
            0, 3, 1, 2), target.unsqueeze(0).permute(0, 3, 1, 2))
        ssim_loss = (1-ssim)

        l2_loss = self.l2(source, target)

        return ssim_weight*ssim_loss + (1-ssim_weight)*l2_loss
        # return self.l2(source, target)

    def step(self, image_info: ImageInfo, ground_truth: torch.Tensor, cover: bool = False, grad: bool = True) -> Tuple[torch.Tensor, dict]:

        # assert ground_truth.device == self.splatter.device

        self.optimizer = torch.optim.Adam(
            self.splatter.gaussians.parameters(), lr=self.lr, betas=(0.9, 0.99))
        self.optimizer.zero_grad()

        if grad:
            render_image, gt_depth = self.splatter(
                image_info, ground_truth, cover)

            # resize gt_depth to match render_image
            if image_info.id not in self.depth_cache:

                gt_depth = resize_image(
                    gt_depth, render_image.shape[1], render_image.shape[0])

                self.depth_cache[image_info.id] = gt_depth
        else:
            with torch.no_grad():
                render_image, gt_depth = self.splatter(
                    image_info, ground_truth, cover)
                return render_image, {}

        render_image_rgb = render_image[..., :3]
        ground_truth = ground_truth.to(self.splatter.device)

        render_depth = render_image[..., 4]
        render_depth = self.depth_normalize(render_depth)
        render_depth = render_depth.unsqueeze(-1)

        gt_depth = self.depth_cache[image_info.id].to(self.splatter.device)
        gt_depth = self.depth_normalize(gt_depth)

        depth_loss = self.critic(render_depth, gt_depth)
        rgb_loss = self.critic(render_image_rgb, ground_truth)

        loss = rgb_loss + depth_loss*10.0

        dump = {
            "loss": loss.item(),
            "count": self.splatter.gaussians.pos.shape[0],
            "lr": self.optimizer.param_groups[0]["lr"],
            # "depth_paramter": self.splatter.depth_paramter.tolist(),
            # "depth": render_depth[0],
        }

        loss.backward()
        self.optimizer.step()

        return render_image, dump

    def eval(self):
        pass


if __name__ == "__main__":
    frame = 0
    downsample = 4
    lr = 0.003
    # lr = 0.0005
    batch = 40
    refine = False
    # refine = True
    test = False
    # test = True

    if test:
        batch = 1

    # dataset = ColmapDataset("dataset/nerfstudio/poster",
    dataset = ColmapDataset("dataset/nerfstudio/stump",
                            # dataset = ColmapDataset("dataset/nerfstudio/aspen",
                            # dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
                            # dataset = ColmapDataset("dataset/nerfstudio/person",
                            downsample_factor=downsample)

    if test:
        trainer = Trainer(ckpt="3dgs_slam_ckpt.pth",
                          # trainer = Trainer(ckpt="3dgs_slam_ckpt_refine.pth",
                          camera=dataset.camera, lr=lr, downsample=downsample)
    else:
        trainer = Trainer(
            camera=dataset.camera, lr=lr, downsample=downsample)

    bar = tqdm(range(0, len(dataset.image_info)))

    window_list = []
    for img_id in bar:
        frame = img_id

        ground_truth = dataset.images[frame]
        ground_truth = ground_truth.to(torch.float).to(
            trainer.splatter.device)/255

        image_info = dataset.image_info[frame]

        if frame % 5 == 0:
            window_list.append([image_info, ground_truth])

            if len(window_list) > batch:
                window_list.pop(0)

        current = image_info.id

        for i in range(2):
            # for info, gt in reversed(window_list):
            for info, gt in (window_list):

                if test:
                    grad = False
                    cover = False
                else:
                    grad = True
                    cover = (i == 0) and (current == info.id)

                render_image, status = trainer.step(image_info=info,
                                                    ground_truth=gt, cover=cover, grad=grad)
                # print(render_image.shape)

                bar.set_postfix(status)
                # save image
        save_image("output.png",  render_image[..., :3])

        depth_image = render_image[..., 4].unsqueeze(-1)
        depth_image = maxmin_normalize(depth_image)
        depth_image = depth_image.repeat(1, 1,  3)
        save_image("depth_output.png", depth_image)

        if not test:
            trainer.splatter.save_ckpt("3dgs_slam_ckpt.pth")
