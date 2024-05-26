
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Union


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: Union[float, np.ndarray]
    image_ids: np.ndarray
    point2D_idxs: np.ndarray
