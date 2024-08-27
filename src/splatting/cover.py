import cv2
import time
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
from utils.function import save_image, resize_image, normalize, maxmin_normalize
import ipdb


class CoverSplatter(Splatter):
    def __init__(self, init_points: Union[Point3D, dict, None] = None,  load_ckpt: Union[str, None] = None, downsample=1, use_sh_coeff=False):

        super(CoverSplatter, self).__init__(
            init_points, load_ckpt, downsample, use_sh_coeff)

        self.distance: int = 8
        self.depth_estimator = Estimator()
        self.down_w, self.down_h = 0, 0
        self.coords: torch.Tensor
        self.depth_paramter = nn.Parameter(torch.tensor([0.08, 5.0, 0.0]))

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

        camera_space *= depth.to(self.device)

        world = torch.einsum('ij,bj->bi', R_inv,
                             camera_space - t)  # Shape: (N, 3)

        return world

    def cover_point(self, image_info: ImageInfo, ground_truth: torch.Tensor, depth: torch.Tensor, render_image: torch.Tensor, alpha_threshold: float = 0.5):

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

        n = uncover_point.shape[0]

        # todo depth base auto scale
        uncover_scale = torch.ones(
            # (uncover_point.shape[0], 3))*self.distance*0.001
            (uncover_point.shape[0], 3))*0.02
        # scale = (1.0/(uncover_depth*0.09+0.001))*4.0
        # scale = uncover_depth/self.camera.fx
        # uncover_scale = torch.ones(
        #     (uncover_point.shape[0], 3))*0.01*self.distance*scale
        new_quaternion = torch.Tensor(
            [1, 0, 0, 0]).unsqueeze(dim=0).repeat(n, 1)
        new_opacity = torch.ones(n)*self.init_opacity

        append_gaussian = Gaussians(
            pos=uncover_point, rgb=uncover_color, opacty=new_opacity, scale=uncover_scale, quaternion=new_quaternion, device=self.device, init_value=True)

        return append_gaussian

    def adaption_control(self, gaussians: Gaussians, grad_threshold=10.0):

        with_grad = not isinstance(gaussians.pos.grad, type(None))
        with_grad = False

        if with_grad:
            add_mask = gaussians.pos.grad > grad_threshold
            add_mask = add_mask.any(-1)

            append_gaussian = gaussians.filte(add_mask)

        # scale < 0.1
        del_mask = torch.norm(gaussians.scale, dim=-1) < 0.01
        # scale > 100.0
        del_mask = torch.logical_or(
            del_mask, torch.norm(gaussians.scale, dim=-1) > 1000.0)
        # opacity < 0.001
        del_mask = torch.logical_or(
            del_mask,  gaussians.opacity < 0.001)

        delete_count = torch.logical_not(del_mask).sum()

        append_count = 0

        gaussians = gaussians.filte(torch.logical_not(del_mask))

        if with_grad:

            add_mask = torch.logical_and(
                add_mask, torch.logical_not(del_mask))

            append_count = len(append_gaussian)
            gaussians.append(append_gaussian)

        return gaussians, (delete_count, append_count)

    def forward(self, image_info: ImageInfo, ground_truth: torch.Tensor, cover: bool = False):

        assert self.camera != None

        render_image = super().forward(image_info)

        if cover:
            assert render_image.shape[:2] == ground_truth.shape[:2]

            gt_depth = self.depth_estimator.estimate(
                ground_truth.cpu().numpy()).cpu()
            # uncover_depth = (uncover_depth-uncover_depth.min())/uncover_depth.max()
            # depth = (1.0/(depth*0.07+0.001))
            # uncover_depth = (1.0/(uncover_depth*0.03+0.001))*5.0
            # uncover_depth = (1.0/(uncover_depth*0.03+0.0001))*2.5
            scaled_depth = (1.0/(gt_depth*0.08+0.001))*3.0
            # scaled_depth = (1.0/(gt_depth*0.08+0.001))*5.0
            # poster
            # uncover_depth = (1.0/(uncover_depth*0.01+0.001))*0.5
            # uncover_depth = (1.0/(uncover_depth*0.08+0.001))*10.0
            # uncover_depth = (1.0/(uncover_depth*0.1+0.0001))
            # uncover_depth = 1.0/uncover_depth*0.01

            # scaled_depth = (
            #     1.0/(gt_depth*self.depth_paramter[0]+0.0001))*self.depth_paramter[1]

            append_gaussian = self.cover_point(
                image_info, ground_truth, scaled_depth, render_image, alpha_threshold=0.7)

            # self.gaussians, status = self.adaption_control(
            #     self.gaussians, grad_threshold=0.01)
            # print(status)

            self.gaussians.append(append_gaussian)
            return render_image, gt_depth

        return render_image, None


if __name__ == "__main__":
    frame = 0
    downsample = 4
    lr = 0.005
    # lr = 0.001
    ssim_weight = 0.1
    batch = 40

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
        depth = None

        for i in range(batch):

            render_image, gt_depth = splatter(
                image_info, ground_truth, i % 10 == 0)
            if gt_depth is not None:
                depth = gt_depth
                depth = resize_image(
                    depth, render_image.shape[1], render_image.shape[0])
                depth = depth.to(splatter.device)
                depth, _ = normalize(depth)
                depth = maxmin_normalize(depth)

            render_rgb = torch.clamp(render_image[..., :3], 0, 1)

            ground_truth = ground_truth.to(splatter.device)

            optimizer = torch.optim.Adam(
                splatter.gaussians.parameters(), lr=lr)

            loss_rgb = l2(render_rgb, ground_truth)

            render_depth = render_image[...,
                                        4].unsqueeze(-1)
            render_depth, _ = normalize(render_depth)
            render_depth = maxmin_normalize(render_depth)

            loss_depth = l2(render_depth, depth)*10.0

            loss = loss_rgb+loss_depth
            dump = {
                "loss": loss.item(),
                "count": splatter.gaussians.pos.shape[0],
                "lr": optimizer.param_groups[0]["lr"],
                "depth_loss": loss_depth.item(),
                # "param": splatter.depth_paramter
                # "grad": splatter.gaussians.pos.grad[0]
            }
            bar.set_postfix(dump)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save image

        # depth = [..., 4]
        # depth = (depth - depth.min())/depth.max()
        # depth = depth.unsqueeze(-1).repeat(1, 1, 3).float()

        # print(depth)
        # save_image("output.png", depth)
        image_info = dataset.image_info[0]
        render_image, gt_depth = splatter(
            image_info, None, False)
        render_depth = render_image[..., 4].unsqueeze(-1)
        render_depth, _ = normalize(render_depth)
        render_depth = maxmin_normalize(render_depth)
        render_depth = render_depth.repeat(1, 1, 3).float()
        image = render_depth
        # image = render_image[:, :, :3]
        save_image("output.png", image)

        splatter.save_ckpt("3dgs_slam_ckpt.pth")
