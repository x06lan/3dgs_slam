
import math
import torch
import torch.nn as nn
import gaussian_cuda

import utils
from renderer import draw, world2camera, global_culling, trun_exp
from gaussian import Gaussians
from tiles import Tiles
import utils.image


class RayInfo:
    def __init__(self, w2c, tran, H, W, focal_x, focal_y):
        self.w2c = w2c
        self.c2w = torch.inverse(w2c)
        self.tran = tran
        self.H = H
        self.W = W
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.rays_o = - self.c2w @ tran

        lefttop_cam = torch.Tensor(
            [(-W/2 + 0.5)/focal_x, (-H/2 + 0.5)/focal_y, 1.0]).to(self.w2c.device)
        dx_cam = torch.Tensor([1./focal_x, 0, 0]).to(self.w2c.device)
        dy_cam = torch.Tensor([0, 1./focal_y, 0]).to(self.w2c.device)
        self.lefttop = self.c2w @ (lefttop_cam - tran)
        self.dx = self.c2w @ dx_cam
        self.dy = self.c2w @ dy_cam


class Splatter(nn.Module):
    def __init__(self, init_point=None,  load_ckpt=None, downsample=2, use_sh_coeff=False, debug=False):
        _device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        assert (_device == 'cuda', 'CUDA is not available')
        self.device = _device
        self.downsample = downsample
        self.default_color = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.gaussians = Gaussians(
            pos=init_point[:, :3].to(_device),
            rgb=init_point[:, 3:6].to(_device),
            opacty=init_point[:, 6].to(_device),
            covariance=init_point[:, 7:].to(_device),
            init_value=True
        )
        if load_ckpt is not None:
            # load checkpoint
            ckpt = torch.load(load_ckpt)
            self.gaussians.pos = nn.Parameter(ckpt["pos"])
            self.gaussians.opacity = nn.Parameter(ckpt["opacity"])
            self.gaussians.rgb = nn.Parameter(ckpt["rgb"])
            self.gaussians.quaternion = nn.Parameter(ckpt["quaternion"])
            self.gaussians.scale = nn.Parameter(ckpt["scale"])

    def set_camera(self, camera):

        self.tile_info = Tiles(
            math.ceil(camera.width),
            math.ceil(camera.height),
            camera.focal_x,
            camera.focal_y,
            self.device
        )
        self.tile_info_cpp = self.tile_info.create_tiles()

    def culling(self):
        pass

    def forward(self, image, imageInfo: utils.image.ImageInfo):
        imageInfo.

        pass
