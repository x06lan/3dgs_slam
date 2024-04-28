
import torch
from typing import Dict, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: torch.tensor
    rgb: torch.tensor
    error: Union[float, torch.tensor]
    image_ids: torch.tensor
    point2D_idxs: torch.tensor
