
import torch
import torch.nn as nn
import gaussian_cuda
import utils
import renderer


class Gaussians(nn.Module):
    def __init__(self, pos, rgb, opacty, scale=None, quaternion=None, covariance=None, init_value=False):
        super().__init__()
        self.init_value = init_value
        if init_value:

            self.pos = pos
            self.rgb = rgb
            self.opacity = opacty
            self.scale = scale if scale is None else nn.parameter.Parameter(
                scale)
            self.quaternion = quaternion if quaternion is None else nn.parameter.Parameter(
                quaternion)
            self.covariance = covariance if covariance is None else nn.parameter.Parameter(
                covariance)

        else:
            self.pos = nn.parameter.Parameter(pos)
            self.rgb = nn.parameter.Parameter(rgb)
            self.opacity = nn.parameter.Parameter(opacty)
            self.scale = nn.parameter.Parameter(scale)
            self.quaternion = nn.parameter.Parameter(quaternion)
            self.covariance = nn.parameter.Parameter(covariance)

    def scale_matrix(self):
        return torch.diag_embed(self.scale)

    def normalize_scale(self):
        scale_activation = "abs"
        if scale_activation == "abs":
            return self.scale.abs()+utils.EPS
        elif scale_activation == "exp":
            return renderer.trunc_exp(self.scale)
        else:
            print("No support scale activation")
            exit()

    def normalize_quaternion(self):
        normed_quat = (self.gaussians.quaternion /
                       self.gaussians.quaternion.norm(dim=1, keepdim=True))
        return normed_quat

    def to_cpp(self):
        _cobj = gaussian_cuda.Gaussian3ds()
        _cobj.pos = self.pos.clone()
        _cobj.rgb = self.rgb.clone()
        _cobj.opa = self.opacity.clone()
        _cobj.cov = self.covariance.clone()
        return _cobj

    def filte(self, mask):
        if self.quaternion is not None and self.scale is not None:
            assert self.covariance is None
            return Gaussians(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opacity[mask],
                quat=self.quaternion[mask],
                scale=self.scale[mask],
            )
        else:
            assert self.covariance is not None
            return Gaussians(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opacity[mask],
                cov=self.covariance[mask],
            )

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
        R = utils.q2r(self.quaternion)
        if scale_activation == "abs":
            _scale = self.scale.abs()+utils.EPS
        elif scale_activation == "exp":
            _scale = renderer.turn_exp(self.scale)
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
        torch.nn.init.uniform_(self.opacity, a=utils.inverse_sigmoid(
            0.01), b=utils.inverse_sigmoid(0.01))


if __name__ == "__main__":
    pass
