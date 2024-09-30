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

import os
import sys
import cv2
import math
import logging
import struct
import functools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


from termcolor import colored
from typing import Dict, Union
from pathlib import Path

from dataclasses import dataclass


from utils.camera import Camera
from utils.image import ImageInfo
from utils.point import Point3D


@dataclass(frozen=True)
class CameraModel:
    model_id: int
    model_name: str
    num_params: int


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file: Union[str, Path]) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(width=width, height=height, fx=params[0], fy=params[1], cx=params[2], cy=params[3], distortParams=params[4:], fps=30)
            # cameras[camera_id] = Camera(
            #     id=camera_id,
            #     model=model_name,
            #     width=width,
            #     height=height,
            #     params=np.array(params),
            # )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file: Union[str, Path]) -> Dict[int, ImageInfo]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = torch.tensor(binary_image_properties[1:5]).to(torch.float32)
            tvec = torch.tensor(binary_image_properties[5:8]).to(torch.float32)
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            # xys = torch.column_stack(
            #     [map(float, x_y_id_s[0::3]),
            #      map(float, x_y_id_s[1::3])]
            # )
            # print(len(list(map(float, x_y_id_s[0::3]))))
            # print(len(list(map(float, x_y_id_s[1::3]))))

            # xys = torch.column_stack(
            #     [tuple(map(float, x_y_id_s[0::3])),
            #      tuple(map(float, x_y_id_s[1::3]))]
            # )
            # xys = np.column_stack(
            #     [tuple(map(float, x_y_id_s[0::3])),
            #      tuple(map(float, x_y_id_s[1::3]))]
            # )
            xys = np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]])
            xys = torch.tensor(xys)

            # point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            point3D_ids = torch.tensor(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ImageInfo(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                # point3D_ids=point3D_ids,
            )

            # print(image_name)
    return images


def read_points3d_binary(path_to_model_file: Union[str, Path]) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def inverse_sigmoid(y):
    if isinstance(y, torch.Tensor):
        return -torch.log(1 / y - 1)
    elif isinstance(y, np.ndarray):
        return -np.log(1 / y - 1)

    return -math.log(1 / y - 1)


def qvec2rot_matrix(qvec):
    temp = [
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ]
    if isinstance(qvec, torch.Tensor):
        return torch.tensor(temp)
    if isinstance(qvec, np.ndarray):
        return np.array(temp)
    return None


def q2r(qvec):
    # qvec B x 4
    qvec = qvec / qvec.norm(dim=1, keepdim=True)
    rot = [
        1 - 2 * qvec[:, 2] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 1] * qvec[:, 2] - 2 * qvec[:, 0] * qvec[:, 3],
        2 * qvec[:, 3] * qvec[:, 1] + 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 1] * qvec[:, 2] + 2 * qvec[:, 0] * qvec[:, 3],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 2] * qvec[:, 3] - 2 * qvec[:, 0] * qvec[:, 1],
        2 * qvec[:, 3] * qvec[:, 1] - 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 2] * qvec[:, 3] + 2 * qvec[:, 0] * qvec[:, 1],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 2] ** 2,
    ]
    rot = torch.stack(rot, dim=1).reshape(-1, 3, 3)
    return rot


def euler_to_quaternion(x, y, z):
    """
    Convert Euler angles (in degrees) to a quaternion vector.

    Parameters:
    - x: Rotation around the x-axis (roll) in degrees.
    - y: Rotation around the y-axis (pitch) in degrees.
    - z: Rotation around the z-axis (yaw) in degrees.

    Returns:
    - A numpy array representing the quaternion vector [q0, q1, q2, q3].
    """
    # Convert degrees to radians
    x_rad = np.radians(x)
    y_rad = np.radians(y)
    z_rad = np.radians(z)

    # Calculate half angles
    cy = np.cos(z_rad * 0.5)
    sy = np.sin(z_rad * 0.5)
    cp = np.cos(y_rad * 0.5)
    sp = np.sin(y_rad * 0.5)
    cr = np.cos(x_rad * 0.5)
    sr = np.sin(x_rad * 0.5)

    # Compute quaternion components
    q0 = cr * cp * cy + sr * sp * sy  # w
    q1 = sr * cp * cy - cr * sp * sy  # x
    q2 = cr * sp * cy + sr * cp * sy  # y
    q3 = cr * cp * sy - sr * sp * cy  # z

    return np.array([q0, q1, q2, q3])


def save_image(path, image):
    if image.max() <= 1.0:
        image = (image * 255).clip(0, 255)

    img_npy = None
    if isinstance(image, torch.Tensor):
        img_npy = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        img_npy = image
    else:
        raise ValueError("Invalid image type")

    img_npy = img_npy.astype(np.uint8)

    cv2.imwrite(path, (img_npy)[..., ::-1])


def maxmin_normalize(v):
    shape = v.shape
    v = v.reshape(-1)

    v = v - v.min()
    v = v / v.max()

    v = v.reshape(shape)
    return v


def normalize(v):
    shape = v.shape
    v = v.reshape(-1)
    mu = v.mean()
    std = v.std()
    v = (v - mu) / std
    v = v.reshape(shape)
    return v, (mu, std)


@functools.lru_cache(maxsize=64)
def resize_image(image, w, h, mode="bilinear"):

    image = image.permute(2, 0, 1).unsqueeze(0)
    # Perform interpolation
    image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)

    # Reshape back to (H, W, C) format
    image = image.squeeze(0).permute(1, 2, 0)

    return image


def convert_z_up_to_y_up(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    quaternion[1] = z
    quaternion[2] = -y
    quaternion[3] = x
    return quaternion


def downsample_imags(dir, downscale):
    os.mkdir(f"{dir}_{downscale}")
    for filename in tqdm(os.listdir(dir)):
        img = cv2.imread(os.path.join(dir, filename))
        img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
        cv2.imwrite(os.path.join(f"{dir}_{downscale}", filename), img)
        # print(os.path.join(dir + "_2", filename))
