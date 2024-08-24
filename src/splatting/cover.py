import cv2
import time
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union, Tuple, Optional
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import kornia
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
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


@functools.lru_cache(maxsize=64)
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
        self.distance: int = 64
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
        R_inv = torch.inverse(extrinsics[:3, :3]).to(
            torch.float32).to(self.device)

        t = extrinsics[:3, 3].unsqueeze(0).to(self.device)

        screen_space = image_coord.clone().to(torch.float32)

        screen_space = torch.cat((screen_space, torch.ones(
            (batch, 1)).to(torch.float32)), dim=1).to(self.device)  # Shape: (N, 4)

        camera_space = torch.einsum(
            'ij,bj->bi', K_inv, screen_space)  # Shape: (N, 3)

        # depth = (1.0/(depth*0.07+0.001))
        depth = (1.0/(depth*0.08+0.001))*5.0

        camera_space *= depth.to(self.device)

        world = torch.einsum('ij,bj->bi', R_inv,
                             camera_space - t)  # Shape: (N, 3)

        return world

    # def screen_space_to_world_coords(self, extrinsics: torch.Tensor, camera: Camera, image_coord: torch.Tensor, depth: torch.Tensor):

    #     K = torch.zeros(4, 4).to(torch.float32).to(self.device)
    #     K[:3, :3] = camera.K_tensor
    #     K[3, 3] = 1.0

    #     E = (extrinsics).to(torch.float32).to(self.device)

    #     w = torch.tensor(camera.width).to(torch.float32).to(self.device)
    #     h = torch.tensor(camera.height).to(torch.float32).to(self.device)

    #     screen_space = image_coord.clone().to(torch.float32)

    #     K = K.unsqueeze(0)
    #     E = E.unsqueeze(0)
    #     w = w.unsqueeze(0)
    #     h = h.unsqueeze(0)

    #     screen_space = screen_space.to(self.device)
    #     depth = depth.to(self.device)

    #     depth = (1.0/(depth*0.09+0.001))*4.0

    #     pinhole = kornia.geometry.camera.PinholeCamera(
    #         K, E, h, w)
    #     return pinhole.unproject(screen_space, depth)

    def cover_point(self, image_info: ImageInfo, ground_truth: torch.Tensor, render_image: torch.Tensor, alpha_threshold: float = 0.5):

        assert render_image.shape[:2] == ground_truth.shape[:2]

        depth = self.depth_estimator.estimate(ground_truth.cpu().numpy()).cpu()

        # resize
        render_image_down = resize_image(
            render_image, self.down_w, self.down_h)
        ground_truth_down = resize_image(
            ground_truth, self.down_w, self.down_h)
        depth_down = resize_image(
            depth, self.down_w, self.down_h)

        # error_threshold = 0.3
        alpha_mask = render_image_down[:, :, 3] < alpha_threshold
        # loss_mask = ((render_image_down[:, :, :3]-ground_truth_down.to(self.device)).pow(2).mean(dim=-1).sqrt()
        #              ) > error_threshold

        mask = alpha_mask
        mask = mask.cpu()

        uncover_coords = self.coords[mask]
        uncover_depth = depth_down[mask]
        uncover_color = ground_truth_down[mask]
        # 0 to 1 to inv sigmoid
        uncover_color = torch.log(uncover_color/(1-uncover_color))

        uncover_point = self.screen_space_to_world_coords(
            image_info.extrinsic(), self.camera, uncover_coords, uncover_depth)

        # todo depth base auto scale
        # scale = (1.0/(uncover_depth*0.09+0.001))*4.0

        uncover_scale = torch.ones(
            (uncover_point.shape[0], 3))*0.02
        # scale = uncover_depth/self.camera.fx
        # uncover_scale = torch.ones(
        #     (uncover_point.shape[0], 3))*0.01*self.distance*scale

        depth = mask.unsqueeze(2).repeat(1, 1, 3).float()
        # depth = render_image[:, :, 3].unsqueeze(2).repeat(1, 1, 3).float()
        # print(depth)
        depth *= render_image_down[:, :, :3].cpu()

        return depth, uncover_point, uncover_color, uncover_scale

    def adaption_control(self, gaussians: Gaussians, grad_threshold=10.0):
        with_grad = not isinstance(gaussians.pos.grad, type(None))

        if with_grad:
            add_mask = gaussians.pos.grad > grad_threshold

            append_gaussian = gaussians.filte(add_mask)

        # scale < 0.1
        del_mask = torch.norm(gaussians.scale, dim=-1) < 0.0001
        # scale > 1.0
        del_mask = torch.logical_or(
            del_mask, torch.norm(gaussians.scale, dim=-1) > 1.0)
        # opacity < 1.0
        del_mask = torch.logical_or(
            del_mask,  gaussians.opacity < 0.05)

        delete_count = torch.logical_not(del_mask).sum()

        append_count = 0

        gaussians = gaussians.filte(torch.logical_not(del_mask))

        if with_grad:

            append_count = append_gaussian.shape[0]
            gaussians.append(append_gaussian)

        return gaussians, (delete_count, append_count)

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

            # self.gaussians, _ = self.adaption_control(
            #     self.gaussians, grad_threshold=10)

            self.gaussians.append(append_gaussian)
            # self.gaussians = append_gaussian
            return render_image, depth

        return render_image, None


if __name__ == "__main__":
    frame = 0
    downsample = 4
    # lr = 0.005
    lr = 0.001
    ssim_weight = 0.1
    batch = 40

    # dataset = ColmapDataset("dataset/nerfstudio/poster",
    dataset = ColmapDataset("dataset/nerfstudio/stump",
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
        save_image("output.png", render_image[:, :, :3])
        splatter.save_ckpt("3dgs_slam_ckpt.pth")
