import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union, Tuple, Optional
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
# from pytorch3d.renderer import PerspectiveCameras


from depth_estimator.estimator import Estimator
from splatter import Splatter
from gaussian import Gaussians
from parser.dataset import ColmapDataset


from utils.image import ImageInfo
from utils.camera import Camera
from utils.point import Point3D
from utils.function import save_image
import ipdb


def resize_image(image, w, h, mode='bilinear'):

    image = image.permute(2, 0, 1).unsqueeze(0)
    # Perform interpolation
    image = F.interpolate(image, size=(
        h, w), mode='bilinear', align_corners=False)

    # Reshape back to (H, W, C) format
    image = image.squeeze(0).permute(1, 2, 0)

    return image


class CoverSplatter(Splatter):
    def __init__(self, init_points: Union[Point3D, dict, None] = None,  load_ckpt: Union[str, None] = None, downsample=1, use_sh_coeff=False):

        super(CoverSplatter, self).__init__(
            init_points, load_ckpt, downsample, use_sh_coeff)
        self.distance: int = 16
        self.depth_estimator = Estimator()
        self.down_w, self.down_h = 0, 0
        self.coords: torch.Tensor

    def set_camera(self, _camera: Camera):

        super().set_camera(_camera)

        self.grid(self.distance)

    def grid(self, distance: int):

        assert self.camera != None

        width, height = self.camera.width, self.camera.height
        self.down_w, self.down_h = int(
            width / distance), int(height/distance)

        x = (torch.arange(self.down_w)*distance +
             0.5*distance).repeat(self.down_h, 1)
        y = (torch.arange(self.down_h)*distance +
             0.5*distance).repeat(self.down_w, 1).t()

        self.coords = torch.stack((x, y), dim=2).to(torch.float32)
        # self.coords = self.coords.reshape(-1, 2)

    def screen_space_to_world_coords(self, extrinsics: torch.Tensor, camera: Camera, image_coord: torch.Tensor, depth: torch.Tensor):

        batch = image_coord.shape[0]

        K_inv = torch.inverse(camera.K_tensor).to(
            torch.float32).to(self.device)
        # E = (extrinsics).to(torch.float32).to(self.device)
        R_inv = torch.inverse(extrinsics[:3, :3]).to(
            torch.float32).to(self.device)
        t = extrinsics[:3, 3].unsqueeze(0).to(self.device)

        screen_space = image_coord.clone().to(torch.float32)

        screen_space *= torch.tensor([camera.width,
                                      camera.height]).to(torch.float32)

        screen_space[:, 0] = (screen_space[:, 0] - camera.cx)
        screen_space[:, 1] = (screen_space[:, 1] - camera.cy)

        screen_space[:, 0] = screen_space[:, 0] / camera.fx*1.07
        screen_space[:, 1] = screen_space[:, 1] / camera.fy*0.65

        screen_space = torch.cat((screen_space, torch.ones(
            (batch, 1)).to(torch.float32)), dim=1).to(self.device)  # Shape: (N, 4)

        camera_space = torch.einsum(
            'ij,bj->bi', K_inv, screen_space)  # Shape: (N, 3)

        camera_space *= depth.to(self.device)

        world = torch.einsum('ij,bj->bi', R_inv,
                             camera_space - t)  # Shape: (N, 3)

        return world

    def cover_point(self, image_info: ImageInfo, ground_truth: torch.Tensor, render_image: torch.Tensor, alpha_threshold: float = 0.5):

        assert render_image.shape[:2] == ground_truth.shape[:2]

        depth = self.depth_estimator.estimate(ground_truth.cpu().numpy()).cpu()

        render_image_down = resize_image(
            render_image, self.down_w, self.down_h)
        ground_truth_down = resize_image(
            ground_truth, self.down_w, self.down_h)
        depth_down = resize_image(
            depth, self.down_w, self.down_h)

        mask = render_image_down[:, :, 3] < alpha_threshold
        mask = mask.cpu()

        uncover_coords = self.coords[mask]
        uncover_depth = depth_down[mask]
        uncover_color = ground_truth_down[mask]

        # depth = (depth-depth.min())/(depth.max()-depth.min())

        uncover_depth = (1.0/(uncover_depth*0.09+0.001))*4.0
        uncover_point = self.screen_space_to_world_coords(
            image_info.extrinsic(), self.camera, uncover_coords, uncover_depth)

        # todo depth base auto scale
        uncover_scale = torch.ones((uncover_point.shape[0], 3))*0.08

        depth = mask.unsqueeze(2).repeat(1, 1, 3).float()
        # depth = render_image[:, :, 3].unsqueeze(2).repeat(1, 1, 3).float()
        # print(depth)
        depth *= render_image_down[:, :, :3].cpu()

        return depth, uncover_point, uncover_color, uncover_scale

    def forward(self, image_info: ImageInfo, ground_truth: torch.Tensor, cover: bool = False):

        assert self.camera != None

        render_image = super().forward(image_info)

        if cover:
            depth, new_point, new_color, new_scale = self.cover_point(
                image_info, ground_truth, render_image, alpha_threshold=0.7)

            n = new_point.shape[0]
            new_quaternion = torch.Tensor(
                [1, 0, 0, 0]).unsqueeze(dim=0).repeat(n, 1)
            new_opacity = torch.ones(n)*self.init_opacity

            append_gaussian = Gaussians(
                pos=new_point, rgb=new_color, opacty=new_opacity, scale=new_scale, quaternion=new_quaternion, device=self.device, init_value=True)

            # del self.gaussians
            # self.gaussians = append_gaussian
            self.gaussians.append(append_gaussian)
            return render_image, depth

        return render_image, None


if __name__ == "__main__":
    frame = 0
    downsample = 4
    # lr = 0.005
    lr = 0.002
    ssim_weight = 0.1
    batch = 20

    dataset = ColmapDataset("dataset/nerfstudio/poster",
                            # dataset = ColmapDataset("dataset/nerfstudio/stump",
                            # dataset = ColmapDataset("dataset/nerfstudio/aspen",
                            # dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
                            # dataset = ColmapDataset("dataset/nerfstudio/person",
                            downsample_factor=downsample)
    # splatter = CoverSplatter(
    #     load_ckpt="3dgslam_ckpt.pth", downsample=downsample)
    splatter = CoverSplatter(downsample=downsample)

    splatter.set_camera(dataset.camera)

    bar = tqdm(range(0, len(dataset.image_info)))

    l2 = nn.MSELoss()

    for img_id in bar:
        frame = img_id
        # frame = img_id % 10

        ground_truth = dataset.images[frame]
        ground_truth = ground_truth.to(torch.float)/255

        # raw_image = ground_truth.numpy()

        image_info = dataset.image_info[frame]

        for i in range(batch):

            render_image, mask = splatter(
                image_info, ground_truth, i % 10 == 0)

            render_image = render_image[..., :3]

            ground_truth = ground_truth.to(splatter.device)

            optimizer = torch.optim.Adam(
                splatter.gaussians.parameters(), lr=lr)
            loss = l2(render_image, ground_truth)
            dump = {
                "loss": loss.item(),
                "count": splatter.gaussians.pos.shape[0],
                "lr": optimizer.param_groups[0]["lr"]
                # "grad": splatter.gaussians.pos.grad[0]
            }
            bar.set_postfix(dump)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save image
        save_image("output.png", render_image)
        splatter.save_ckpt("3dgslam_ckpt.pth")
