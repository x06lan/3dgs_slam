import math
import numpy as np
import torch
import torch.nn as nn
import gaussian_cuda
import ipdb
import cv2
import time
from tqdm import tqdm
from typing import Union, Tuple
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from pykdtree.kdtree import KDTree


import utils.function as function
import gaussian_cuda
from renderer import draw, global_culling
from gaussian import Gaussians
from tiles import Tiles
from parser.dataset import ColmapDataset
from utils.image import ImageInfo
from utils.camera import Camera
from utils.point import Point3D


class Splatter(nn.Module):
    def __init__(self, init_points: Union[Point3D, None] = None,  load_ckpt: Union[str, None] = None, downsample=1, use_sh_coeff=False, debug=False):

        super().__init__()

        self.device = torch.device("cuda")
        self.downsample: int = downsample
        self.debug: bool = debug
        self.near: float = 0.3
        self.default_color = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.init_scale = 0.1
        self.init_opacity = 0.8
        self.camera: Camera = None
        # self.w2c_r = None
        # self.w2c_t = None
        self.use_sh_coeff: bool = False
        self.render_weight_normalize: bool = False
        self.fast_drawing: bool = False
        self.gaussians: Gaussians = None

        # TODO : corrent this initialization

        if (init_points is not None):
            _rgb = []
            _pos = []
            for pid, point in init_points.items():
                _pos.append(torch.from_numpy(point.xyz))
                _rgb.append(function.inverse_sigmoid(
                    torch.from_numpy(point.rgb)))

            rgb = torch.stack(_rgb).to(torch.float32).to(self.device)
            pos = torch.stack(_pos).to(torch.float32).to(self.device)

            _pos_np = pos.cpu().numpy()
            kd_tree = KDTree(_pos_np)
            dist, idx = kd_tree.query(_pos_np, k=4)
            mean_min_three_dis = dist[:, 1:].mean(axis=1)
            mean_min_three_dis = torch.Tensor(mean_min_three_dis).to(
                torch.float32) * self.init_scale

            self.gaussians = Gaussians(
                pos=pos,
                rgb=rgb,
                opacty=torch.ones(len(init_points)).to(self.device),
                quaternion=torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(
                    len(init_points), 1).to(torch.float32).to(self.device),  # B x 4
                scale=torch.ones(len(init_points), 3).to(torch.float32).to(
                    self.device)*mean_min_three_dis.unsqueeze(dim=1).to(self.device),
                init_value=True
            )
        elif load_ckpt is not None:
            # load checkpoint

            ckpt = torch.load(load_ckpt)
            # ipdb.set_trace()
            self.gaussians = Gaussians(
                pos=ckpt["pos"],

                rgb=ckpt["rgb"],
                # rgb=torch.ones(
                #     ckpt["pos"].shape[0], 3).to(torch.float32).to(self.device)*0.1,
                opacty=ckpt["opa"],
                # opacty=torch.ones(
                #     ckpt["pos"].shape[0]).to(torch.float32).to(self.device)*0.3,
                quaternion=ckpt["quat"],  # B x 4
                scale=ckpt["scale"],
                init_value=True
            )
            for key, value in ckpt.items():
                print(key, value.shape)

            # self.gaussians.rgb = torch.ones(
            #     ckpt["pos"].shape[0], 3).to(torch.float32).to(self.device)*0.5

            # self.gaussians.pos = nn.Parameter(ckpt["pos"])
            # # self.gaussians.opacity = nn.Parameter(ckpt["opacity"])
            # self.gaussians.opacity = nn.Parameter(ckpt["opa"])
            # self.gaussians.rgb = nn.Parameter(ckpt["rgb"])
            # # self.gaussians.quaternion = nn.Parameter(ckpt["quaternion"])
            # self.gaussians.quaternion = nn.Parameter(ckpt["quat"])
            # self.gaussians.scale = nn.Parameter(ckpt["scale"])

    def save_ckpt(self, path):
        ckpt = {
            "pos": self.gaussians.pos,
            "opa": self.gaussians.opacity,
            "rgb": self.gaussians.rgb,
            "quat": self.gaussians.quaternion,
            "scale": self.gaussians.scale,
        }
        torch.save(ckpt, path)

    def w2c(self, img_info: ImageInfo):

        w2c_q = img_info.qvec
        w2c_t = img_info.tvec.to(self.device)

        w2c_r = function.qvec2rot_matrix(w2c_q).squeeze().to(
            torch.float32)
        return w2c_r.to(self.device), w2c_t.to(self.device)

    def RayInfo(self, tile_info: Tiles, w2c_r: torch.tensor, w2c_t: torch.tensor):

        # invert of world to camera
        # c2w = torch.inverse(world_camera.matrix()).to(tile_info.device)
        c2w = torch.inverse(w2c_r)

        rays_o = -c2w @ w2c_t
        W = tile_info.padded_width
        H = tile_info.padded_height
        focal_x = tile_info.focal_x
        focal_y = tile_info.focal_y

        lefttop_cam = torch.Tensor(
            [(-W/2 + 0.5)/focal_x,
             (-H/2 + 0.5)/focal_y,
             1.0]).to(tile_info.device)

        dx_cam = torch.Tensor([1.0 / focal_x, 0, 0]).to(w2c_r.device)
        dy_cam = torch.Tensor([0, 1.0/focal_y, 0]).to(w2c_r.device)
        lefttop = c2w @ (lefttop_cam - w2c_t)
        dx = c2w @ dx_cam
        dy = c2w @ dy_cam

        return rays_o, lefttop, dx, dy

    def set_camera(self, _camera: Camera):

        self.camera = _camera
        self.tile_info = Tiles(
            math.ceil(_camera.width/self.downsample),
            math.ceil(_camera.height/self.downsample),
            _camera.fx/self.downsample,
            _camera.fy/self.downsample,
            self.device
        )

        self.tile_info_cpp = self.tile_info.create_tiles()

    def project_culling(self, w2c_r: torch.tensor, w2c_t: torch.tensor):

        magic_number = 1.2

        # self.gaussians.normalize_quaternion(),
        # self.gaussians.normalize_scale(),
        half_width = self.camera.width*magic_number/2/self.camera.fx
        half_height = self.camera.height*magic_number/2/self.camera.fy

        # 2d position,2d covariance, mask
        _pos, _cov, mask = global_culling(
            self.gaussians.pos,
            self.gaussians.normalize_quaternion(),
            self.gaussians.normalize_scale(),
            w2c_r.detach(),
            w2c_t.detach(),
            self.near,
            half_width,
            half_height
        )
        mask = mask.bool()

        _rgb = self.gaussians.rgb[mask]

        if not self.use_sh_coeff:
            _rgb = _rgb.sigmoid()
        culled_gaussians = Gaussians(
            pos=_pos[mask],
            covariance=_cov[mask],
            rgb=_rgb,
            opacty=self.gaussians.opacity[mask].sigmoid(),
            # quaternion=self.gaussians.quaternion[mask],
            # scale=self.gaussians.scale[mask].sigmoid(),
        )

        return culled_gaussians, mask

    def render(self, culling_gaussians: Gaussians,  w2c_r: torch.tensor, w2c_t: torch.tensor):
        # if len(gaussians.pos) == 0:
        #     return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

        # print("pos", culling_gaussians.pos.shape)

        # print("rgb", culling_gaussians.rgb.shape)
        # print(culling_gaussians.rgb)

        tile_n_point = torch.zeros(
            len(self.tile_info), device=self.device, dtype=torch.int32)
        # MAXP = len(self.culling_gaussian_3d_image_space.pos)//10
        tile_max_point = len(culling_gaussians.pos)//20

        tile_gaussian_list = torch.ones(
            len(self.tile_info), tile_max_point, device=self.device, dtype=torch.int32) * -1

        tile_culling_method = "prob2"
        method_config = {"dist": 0, "prob": 1, "prob2": 2}
        tile_culling_dist_thresh = 0.5
        if tile_culling_method != "dist":
            tile_culling_dist_thresh = tile_culling_dist_thresh**2

        # culling tiles
        gaussian_cuda.calc_tile_list(
            culling_gaussians.to_cpp(),
            self.tile_info_cpp,
            tile_n_point,
            tile_gaussian_list,
            (self.tile_info.tile_geo_length_x /
             tile_culling_dist_thresh),
            method_config[tile_culling_method],
            self.tile_info.tile_geo_length_x,
            self.tile_info.tile_geo_length_y,
            self.tile_info.n_tile_x,
            self.tile_info.n_tile_y,
            self.tile_info.leftmost,
            self.tile_info.topmost,
        )

        tile_n_point = torch.min(
            tile_n_point, torch.ones_like(tile_n_point)*tile_max_point)

        # if tile_n_point.sum() == 0:
        #     return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

        gathered_list = torch.empty(
            tile_n_point.sum(), dtype=torch.int32, device=self.device)

        tile_ids_for_points = torch.empty(
            tile_n_point.sum(), dtype=torch.int32, device=self.device)

        tile_n_point_accum = torch.cat([torch.Tensor([0]).to(
            self.device), torch.cumsum(tile_n_point, 0)]).to(tile_n_point)
        max_points_for_tile = tile_n_point.max().item()

        # gather culled tiles
        gaussian_cuda.gather_gaussians(
            tile_n_point_accum,
            tile_gaussian_list,
            gathered_list,
            tile_ids_for_points,
            int(max_points_for_tile),
        )
        tile_gaussians = culling_gaussians.filte(
            gathered_list.long())

        # sort by tile id and depth
        BASE = tile_gaussians.pos[..., 2].max()
        id_and_depth = tile_gaussians.pos[..., 2].to(
            torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
        _, sort_indices = torch.sort(id_and_depth)
        tile_gaussians = tile_gaussians.filte(sort_indices)

        # render tiles
        rays_o, lefttop, dx, dy = self.RayInfo(
            self.tile_info, w2c_r, w2c_t)

        rendered_image = draw(
            tile_gaussians.pos,
            tile_gaussians.rgb,
            tile_gaussians.opacity,
            tile_gaussians.covariance,
            tile_n_point_accum,
            self.tile_info.padded_height,
            self.tile_info.padded_width,
            self.tile_info.focal_x,
            self.tile_info.focal_y,
            self.render_weight_normalize,
            False,
            self.use_sh_coeff,
            self.fast_drawing,
            rays_o,
            lefttop,
            dx,
            dy,
        )

        return rendered_image

    def save_image(path, image):
        img_npy = image.clip(0, 1).detach().cpu().numpy()

        cv2.imwrite(
            path, (img_npy*255).astype(np.uint8)[..., ::-1])

    def forward(self,  imageInfo: ImageInfo):
        w2c_r, w2c_t = self.w2c(imageInfo)

        culling_gaussians, mask = self.project_culling(w2c_r, w2c_t)
        padded_rendered_image = self.render(culling_gaussians, w2c_r, w2c_t)

        padded_render_image = torch.clamp(padded_rendered_image, 0, 1)
        render_image = self.tile_info.crop(padded_render_image)

        return render_image


if __name__ == "__main__":

    frame = 0
    downsample = 2
    dataset = ColmapDataset("dataset/nerfstudio/poster",
                            downsample_factor=downsample)
    # dataset = ColmapDataset("dataset/nerfstudio/aspen")

    # splatter = Splatter(init_points=dataset.points3d,
    #                     downsample=1, use_sh_coeff=False)
    splatter = Splatter(load_ckpt="ckpt3.pth", downsample=downsample)
    splatter.set_camera(dataset.camera)

    optimizer = torch.optim.Adam(splatter.gaussians.parameters(), lr=0.003)
    ssim = StructuralSimilarityIndexMeasure(
        reduction="elementwise_mean").to(splatter.device)
    ssim_weight = 0.1

    for img_id in tqdm(range(0, 100)):
        render_image = splatter.forward(dataset.image_info[frame])
        # render_image = (render_image*255).dtype(np.uint8)

        ground_truth = dataset.images[frame].to(splatter.device)
        ground_truth = ground_truth.type_as(render_image)/255

        l1_loss = (ground_truth - render_image).abs().mean()
        ssim_loss = ssim(render_image.unsqueeze(0).permute(
            0, 3, 1, 2), ground_truth.unsqueeze(0).permute(0, 3, 1, 2))

        loss = l1_loss*(1.0-ssim_weight) + ssim_loss*ssim_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        print(splatter.gaussians.rgb.grad.abs().mean())

        # save image
        Splatter.save_image("output.png", render_image)

        # time.sleep(0.3)
