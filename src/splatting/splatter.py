
import math
import torch
import torch.nn as nn
import gaussian_cuda
import transforms as tf

import utils
import gaussian_cuda
from renderer import draw, global_culling
from gaussian import Gaussians
from tiles import Tiles
from utils.image import ImageInfo
from utils.camera import Camera


class Splatter(nn.Module):
    def __init__(self, init_point=None,  load_ckpt=None, downsample=1, use_sh_coeff=False, debug=False):
        _device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        assert (_device == 'cuda', 'CUDA is not available')
        self.device = _device
        self.downsample: int = downsample
        self.debug: bool = debug
        self.near: float = 0.3
        self.default_color = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.camera: Camera = None
        self.w2c_r = None
        self.w2c_t = None
        self.use_sh_coeff: bool = False

        # TODO : corrent this initialization
        self.gaussians: Gaussians = Gaussians(
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

    def RayInfo(self, tile_info: Tiles, img_info: ImageInfo):
        world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(img_info.qvec), img_info.tvec,
        )
        w2c_q = torch.from_numpy(world_camera.rotation().wxyz).to(
            torch.float32).to(tile_info.device)

        w2c_t = torch.from_numpy(world_camera.translation()).to(
            torch.float32).to(tile_info.device)

        w2c_r = (w2c_q[-1].unsqueeze(0)
                 ).squeeze().to(torch.float32).to(tile_info.device)

        self.w2c_r = w2c_r
        self.w2c_t = w2c_t

        # invert of world to camera
        # c2w = torch.inverse(world_camera.matrix()).to(tile_info.device)
        c2w = torch.inverse(w2c_r.matrix()).to(tile_info.device)

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
            _camera.focal_x/self.downsample,
            _camera.focal_y/self.downsample,
            self.device
        )

        self.tile_info_cpp = self.tile_info.create_tiles()

    def project_culling(self):

        magic_number = 1.2

        # 2d position,2d covariance, mask
        _pos, _cov, mask = global_culling(
            self.gaussians.pos,
            self.gaussians.normalize_quaternion(),
            self.gaussians.normalize_scale(),
            self.w2c_r.detach(),
            self.w2c_t.detach(),
            self.near,
            self.camera.width*magic_number/2.0/self.camera.focal_x,
            self.camera.height*magic_number/2.0/self.camera.focal_y,
        )
        mask = mask.bool()
        _rgb = self.gaussians.rgb[mask]
        if self.use_sh_coeff:
            _rgb = _rgb[mask].sigmoid()

        culled_gaussians = Gaussians(
            pos=_pos[mask],
            covariance=_cov[mask],
            rgb=_rgb,
            opacty=self.gaussians.opacity[mask],
            quaternion=self.gaussians.quaternion[mask],
            scale=self.gaussians.scale[mask].sigmoid(),
        )

        return culled_gaussians, mask

    def render(self, gaussians: Gaussians):
        if len(gaussians.pos) == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

        tile_n_point = torch.zeros(
            len(self.tile_info), device=self.device, dtype=torch.int32)
        # MAXP = len(self.culling_gaussian_3d_image_space.pos)//10
        tile_max_point = len(gaussians.pos)//20

        tile_gaussian_list = torch.ones(
            len(self.tile_info), tile_max_point, device=self.device, dtype=torch.int32) * -1

        tile_culling_method = "prob2"
        method_config = {"dist": 0, "prob": 1, "prob2": 2}
        tile_culling_dist_thresh = 0.1
        if tile_culling_method != "dist":
            tile_culling_dist_thresh = tile_culling_dist_thresh**2

        # culling tiles
        gaussian_cuda.calc_tile_list(
            gaussians.to_cpp(),
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

        if tile_n_point.sum() == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

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
        self.tile_gaussians = gaussians.filte(
            gathered_list.long())
        self.n_tile_gaussians = len(self.tile_gaussians.pos)
        self.n_gaussians = len(gaussians.pos)

        # sort by tile id and depth
        BASE = gaussians.pos[..., 2].max()
        id_and_depth = gaussians.pos[..., 2].to(
            torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
        _, sort_indices = torch.sort(id_and_depth)
        gaussians = gaussians.filte(sort_indices)

        # render tiles
        rendered_image = draw(
            self.tile_gaussians.pos,
            self.tile_gaussians.rgb,
            self.tile_gaussians.opa,
            self.tile_gaussians.cov,
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
            dy,
        )

    def forward(self, image, imageInfo: utils.image.ImageInfo):

        # self.set_camera(Camera(imageInfo))
        rays_o, lefttop, dx, dy = self.RayInfo(self.tile_info, imageInfo)

        gaussians, mask = self.project_culling()
        rendered_image = self.render(gaussians)
        pass
