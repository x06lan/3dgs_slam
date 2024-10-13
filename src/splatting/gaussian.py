
import torch
import torch.nn as nn
import gaussian_cuda

from .renderer import trunc_exp
from utils import function

EPS = 1e-6


class Gaussians(nn.Module):
    def __init__(self, pos, rgb, opacty, scale=None, quaternion=None, covariance=None, init_value=False, device=None):

        super(Gaussians, self).__init__()

        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = device

        if init_value:

            def setup(v): return v.to(torch.float32).to(self.device)

            self.pos = nn.parameter.Parameter(setup(pos))
            self.rgb = nn.parameter.Parameter(setup(rgb))
            self.opacity = nn.parameter.Parameter(setup(opacty))
            self.scale = scale if scale is None else nn.parameter.Parameter(setup(
                scale))
            self.quaternion = quaternion if quaternion is None else nn.parameter.Parameter(setup(
                quaternion))
            self.covariance = covariance if covariance is None else nn.parameter.Parameter(setup(
                covariance))
        else:
            self.pos = pos
            self.rgb = rgb
            self.opacity = opacty
            self.scale = scale
            self.quaternion = quaternion
            self.covariance = covariance

    @property
    def shape(self):
        return self.pos.shape[:2]

    def __len__(self):
        return self.pos.shape[0]

    def append(self, other):
        assert self.quaternion is not None and self.scale is not None
        self.pos = nn.parameter.Parameter(
            torch.cat([self.pos, other.pos]), requires_grad=True)
        self.rgb = nn.parameter.Parameter(torch.cat([self.rgb, other.rgb]))
        self.opacity = nn.parameter.Parameter(
            torch.cat([self.opacity, other.opacity]))
        self.scale = nn.parameter.Parameter(
            torch.cat([self.scale, other.scale]))

        self.quaternion = nn.parameter.Parameter(torch.cat(
            [self.quaternion, other.quaternion]))

    def scale_matrix(self):
        assert self.scale is not None
        return torch.diag_embed(torch.abs(self.scale))

    def normalize_scale(self):
        assert self.scale is not None

        scale_activation = "abs"
        if scale_activation == "abs":
            return self.scale.abs()+EPS
        elif scale_activation == "exp":
            return trunc_exp(self.scale)
        else:
            print("No support scale activation")
            exit()

    def normalize_quaternion(self):
        assert self.quaternion is not None
        normed_quat = (self.quaternion /
                       self.quaternion.norm(dim=1, keepdim=True))
        return normed_quat

    def to_cpp(self):
        _cobj = gaussian_cuda.Gaussian3ds()
        _cobj.pos = self.pos.clone()
        _cobj.rgb = self.rgb.clone()
        _cobj.opa = self.opacity.clone()
        _cobj.cov = self.covariance.clone()
        return _cobj

    def filte(self, mask, init_value=False):

        if self.covariance is not None:
            return Gaussians(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opacty=self.opacity[mask],
                covariance=self.covariance[mask],
                init_value=init_value
            )
        elif self.quaternion is not None and self.scale is not None:
            return Gaussians(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opacty=self.opacity[mask],
                quaternion=self.quaternion[mask],
                scale=self.scale[mask],
                init_value=init_value
            )
        else:
            raise ValueError("No support filter")

    def to(self, *args, **kwargs):
        self.pos.to(*args, **kwargs)
        self.rgb.to(*args, **kwargs)
        self.opacity.to(*args, **kwargs)
        if self.quaternion is not None:
            self.quaternion.to(*args, **kwargs)
        if self.scale is not None:
            self.scale.to(*args, **kwargs)
        if self.covariance is not None:
            self.covariance.to(*args, **kwargs)

    def get_gaussian_3d_cov(self, scale_activation="abs"):

        assert self.quaternion is not None and self.scale is not None

        R = function.q2r(self.quaternion)
        if scale_activation == "abs":
            _scale = self.scale.abs()+EPS
        elif scale_activation == "exp":
            _scale = trunc_exp(self.scale)
        else:
            print("No support scale activation")
            exit()
        # _scale = trunc_exp(self.scale)
        # _scale = torch.clamp(_scale, min=1e-4, max=0.1)
        S = torch.diag_embed(_scale)
        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0, 2, 1))
        return RSSR

    def reset_opacity(self):
        torch.nn.init.uniform_(self.opacity, a=function.inverse_sigmoid(
            0.01), b=function.inverse_sigmoid(0.01))
