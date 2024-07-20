
import torch

from dataclasses import dataclass

import utils
import utils.function


@dataclass(frozen=True)
class BaseImageData:
    id: int
    qvec: torch.tensor
    tvec: torch.tensor
    camera_id: int
    name: str
    xys: torch.tensor
    # point3D_ids: torch.tensor


# @dataclass(frozen=True)
class ImageInfo(BaseImageData):
    def __init__(self, id, qvec, tvec, camera_id, name, xys):
        super().__init__(id, qvec, tvec, camera_id, name, xys)
        pass

    def w2c(self, device="cuda"):

        w2c_q = self.qvec
        w2c_t = self.tvec

        w2c_r = utils.function.qvec2rot_matrix(w2c_q).squeeze().to(
            torch.float32)

        return w2c_r.to(device), w2c_t.to(device)

    def extrinsic(self):
        r, t = self.w2c()
        m = torch.zeros(4, 4)
        m[:3, :3] = r
        m[:3, 3] = t
        m[3, 3] = 1
        return m

    # def rotation_matrix(self):
    #     return utils.qvec2rotmat(self.qvec)

    # def translation(self):
    #     return self.tvec
    # def qvec2rotmat(self):
    #     return qvec2rotmat(self.qvec)
