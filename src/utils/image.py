
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageData:
    id: int
    qvec: torch.tensor
    tvec: torch.tensor
    camera_id: int
    name: str
    xys: torch.tensor
    point3D_ids: torch.tensor


class ImageInfo(ImageData):
    pass
    # def qvec2rotmat(self):
    #     return qvec2rotmat(self.qvec)
