
import torch

from dataclasses import dataclass

import utils


@dataclass(frozen=True)
class BaseImageData:
    id: int
    qvec: torch.tensor
    tvec: torch.tensor
    camera_id: int
    name: str
    xys: torch.tensor
    # point3D_ids: torch.tensor


class ImageInfo(BaseImageData):
    def __init__(self, id, qvec, tvec, camera_id, name, xys):
        super().__init__(id, qvec, tvec, camera_id, name, xys)
        pass
    #     self.rotation_matrix = utils.qvec2rotmat(self.qvec)
    #     self.translation_vector = self.tvec

    # def rotation_matrix(self):
    #     return utils.qvec2rotmat(self.qvec)

    # def translation(self):
    #     return self.tvec
    # def qvec2rotmat(self):
    #     return qvec2rotmat(self.qvec)
