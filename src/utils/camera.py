"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

# import numpy as np
# import cv2
# import g2o
# from utils.utils_geom import add_ones
import torch
import numpy as np
import cv2

# from utils.utils import add_ones


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy, distortParams, fps):
        self.width: int = width
        self.height: int = height
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy
        self.fps: float = fps
        # [k1, k2, p1, p2, k3]
        self.distortParams = distortParams
        self.isDistorted = np.linalg.norm(self.distortParams) > 1e-10
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])
        self.Kinv = np.array([[1/self.fx,       0, -self.cx/self.fx],
                              [0, 1/self.fy, -self.cy/self.fy],
                              [0,        0,                1]])

    @property
    def K_tensor(self):
        return torch.from_numpy(self.K)

    @property
    def Kinv_tensor(self):
        return torch.from_numpy(self.Kinv)

    def project(self, xcs):
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy
        projs = self.K @ xcs.T
        zs = projs[-1]
        projs = projs[:2] / zs
        return projs.T, zs

    # unproject 2D point uv (pixels on image plane) on
    def unproject(self, uv):
        x = (uv[0] - self.cx)/self.fx
        y = (uv[1] - self.cy)/self.fy
        return x, y

    # in:  uvs [Nx2]
    # out: xcs array [Nx3] of normalized coordinates
    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]

    # in:  uvs [Nx2]
    # out: uvs_undistorted array [Nx2] of undistorted coordinates
    def undistort_points(self, uvs):
        if self.isDistorted:
            # uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)   # =>  Error: while undistorting the points error: (-215:Assertion failed) src.isContinuous()
            if isinstance(uvs, torch.Tensor):
                uvs_contiguous = (
                    uvs[:, :2]).contiguos().reshape((uvs.shape[0], 1, 2))
            else:
                uvs_contiguous = np.ascontiguousarray(
                    uvs[:, :2]).reshape((uvs.shape[0], 1, 2))

            uvs_contiguous = (
                uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(
                uvs_contiguous, self.K, self.distortParams, None, self.K)

            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs

    # update image bounds
    def undistort_image_bounds(self):
        uv_bounds = np.ndarray([[self.u_min, self.v_min],
                                [self.u_min, self.v_max],
                                [self.u_max, self.v_min],
                                [self.u_max, self.v_max]], dtype=torhc.float32).reshape(4, 2)
        # print('uv_bounds: ', uv_bounds)
        if self.is_distorted:
            uv_bounds_undistorted = cv2.undistortPoints(
                np.expand_dims(uv_bounds, axis=1), self.K, self.distortParams, None, self.K)
            uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(
                uv_bounds_undistorted.shape[0], 2)
        else:
            uv_bounds_undistorted = uv_bounds
        # print('uv_bounds_undistorted: ', uv_bounds_undistorted)
        self.u_min = min(
            uv_bounds_undistorted[0][0], uv_bounds_undistorted[1][0])
        self.u_max = max(
            uv_bounds_undistorted[2][0], uv_bounds_undistorted[3][0])
        self.v_min = min(
            uv_bounds_undistorted[0][1], uv_bounds_undistorted[2][1])
        self.v_max = max(
            uv_bounds_undistorted[1][1], uv_bounds_undistorted[3][1])


def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# turn [[x,y]] -> [[x,y,1]]


def add_ones_1D(x):
    # return np.concatenate([x,np.array([1.0])], axis=0)
    # return torch.tensor([x[0], x[1], 1])
    return np.ndarray([x[0], x[1], 1])
