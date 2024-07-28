import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union, Tuple, Optional
# from pytorch3d.renderer import PerspectiveCameras


from depth_estimator.estimator import Estimator
from splatter import Splatter
from gaussian import Gaussians
from parser.dataset import ColmapDataset
from torchvision import transforms
from utils.image import ImageInfo
from utils.camera import Camera
from utils.point import Point3D
import ipdb


def resize_image(image, w, h):

    image = image.permute(2, 0, 1).unsqueeze(0)
    # Perform interpolation
    image = F.interpolate(image, size=(
        h, w), mode='bilinear', align_corners=False)

    # Reshape back to (H, W, C) format
    image = image.squeeze(0).permute(1, 2, 0)

    return image


class EMA():
    def __init__(self, beta):
        self.beta = beta
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = self.beta * self.average + (1-self.beta) * value

    def get(self):
        return self.average


class CoverSplatter(Splatter):
    def __init__(self, init_points: Union[Point3D, dict, None] = None,  load_ckpt: Union[str, None] = None, downsample=1, use_sh_coeff=False):

        super(CoverSplatter, self).__init__(
            init_points, load_ckpt, downsample, use_sh_coeff)
        self.distance = 20
        self.depth_estimator = Estimator()
        self.cover_w, self.cover_h = 0, 0

    def set_camera(self, _camera: Camera):

        super().set_camera(_camera)

        width, height = _camera.width, _camera.height
        self.down_w, self.down_h = int(
            width / self.distance), int(height/self.distance)

        x = (torch.arange(self.down_w)*self.distance +
             0.5*self.distance).repeat(self.down_h, 1)
        y = (torch.arange(self.down_h)*self.distance +
             0.5*self.distance).repeat(self.down_w, 1).t()

        self.coords = torch.stack((x, y), dim=2).to(torch.float32)
        # self.coords = self.coords.reshape(-1, 2)

    def screen_space_to_world_coords(self, extrinsics: torch.Tensor, camera: Camera, image_coord: torch.Tensor, depth: torch.Tensor):

        batch = image_coord.shape[0]

        K_inv = torch.inverse(camera.K_tensor).to(torch.float32)
        E = (extrinsics).to(torch.float32)
        R_inv = torch.inverse(extrinsics[:3, :3]).to(torch.float32)
        t = extrinsics[:3, 3].unsqueeze(0)

        screen_space = image_coord.clone().to(torch.float32)

        screen_space *= torch.tensor([camera.width,
                                      camera.height]).to(torch.float32)

        screen_space[:, 0] = (screen_space[:, 0] - camera.cx)
        screen_space[:, 1] = (screen_space[:, 1] - camera.cy)

        screen_space[:, 0] = screen_space[:, 0] / camera.fx*1.07
        screen_space[:, 1] = screen_space[:, 1] / camera.fy*0.65

        screen_space = torch.cat((screen_space, torch.ones(
            (batch, 1)).to(torch.float32)), dim=1)  # Shape: (N, 4)

        camera_space = torch.einsum(
            'ij,bj->bi', K_inv, screen_space)  # Shape: (N, 3)

        camera_space *= depth

        world = torch.einsum('ij,bj->bi', R_inv,
                             camera_space - t)  # Shape: (N, 3)

        return world

    def cover_point(self, image_info: ImageInfo, ground_truth: torch.Tensor, render_image: torch.Tensor, alph_threshold: float = 0.5):

        assert render_image.shape[:2] == ground_truth.shape[:2]

        depth = self.depth_estimator.estimate(ground_truth.numpy()).cpu()

        render_image_down = resize_image(
            render_image, self.down_w, self.down_h)
        ground_truth_down = resize_image(
            ground_truth, self.down_w, self.down_h)
        depth_down = resize_image(depth, self.down_w, self.down_h)

        mask = render_image_down[:, :, 3] < alph_threshold
        mask = mask.cpu()

        uncover_coords = self.coords[mask]
        uncover_depth = depth_down[mask]
        uncover_color = ground_truth_down[mask]

        # depth = (depth-depth.min())/(depth.max()-depth.min())

        # todo depth auto scale
        uncover_depth = (1.0/(uncover_depth*0.09+0.001))*3
        uncover_point = self.screen_space_to_world_coords(
            image_info.extrinsic(), self.camera, uncover_coords, uncover_depth)

        uncover_scale = torch.ones((uncover_point.shape[0], 3))

        return depth, uncover_point, uncover_color, uncover_scale

    def control_points(self, new_point: torch.Tensor, new_color: torch.Tensor, new_scale: torch.Tensor):

        pass

    def forward(self, image_info: ImageInfo, ground_truth: torch.Tensor):

        render_image = super().forward(image_info)

        depth, new_point, new_color, new_scale = self.cover_point(
            image_info, ground_truth, render_image, alph_threshold=0.5)

        n = new_point.shape[0]
        new_quaternion = torch.Tensor(
            [1, 0, 0, 0]).unsqueeze(dim=0).repeat(n, 1)
        new_opacity = torch.ones((n, 1))*0.8

        append_gaussian = Gaussians(
            pos=new_point, rgb=new_color, opacty=new_opacity, scale=new_scale, quaternion=new_quaternion, device=self.device)
        self.gaussians.append(append_gaussian)

        # del append_gaussian

        return render_image


if __name__ == "__main__":
    frame = 0
    downsample = 4
    # device = "cuda"
    dataset = ColmapDataset("dataset/nerfstudio/poster",
                            # dataset = ColmapDataset("dataset/nerfstudio/stump",
                            # dataset = ColmapDataset("dataset/nerfstudio/aspen",
                            # dataset = ColmapDataset("dataset/nerfstudio/redwoods2",
                            # dataset = ColmapDataset("dataset/nerfstudio/person",
                            downsample_factor=downsample)
    splatter = CoverSplatter(load_ckpt="ckpt3.pth", downsample=downsample)
    splatter.set_camera(dataset.camera)

    optimizer = torch.optim.Adam(splatter.gaussians.parameters(), lr=0.003)
    # ssim = StructuralSimilarityIndexMeasure(
    #     reduction="elementwise_mean").to(splatter.device)
    ssim_weight = 0.1

    for img_id in tqdm(range(0, len(dataset.image_info))):
        frame = img_id

        ground_truth = dataset.images[frame]
        ground_truth = ground_truth.to(torch.float)/255

        raw_image = ground_truth.numpy()

        image_info = dataset.image_info[frame]
        render_image = splatter(image_info, ground_truth)

        loss = (ground_truth - render_image).abs().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss)
        # print(splatter.gaussians.rgb.grad.abs().mean())

        # save image
        # Splatter.save_image("output.png", render_image)
