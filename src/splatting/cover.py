import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from pytorch3d.renderer import PerspectiveCameras


from depth_estimator.estimator import Estimator
from splatter import Splatter
from parser.dataset import ColmapDataset
from torchvision import transforms
from utils.image import ImageInfo
import ipdb


def resize_image(image, w, h):

    image = image.permute(2, 0, 1).unsqueeze(0)
    # Perform interpolation
    image = F.interpolate(image, size=(
        h, w), mode='bilinear', align_corners=False)

    # Reshape back to (H, W, C) format
    image = image.squeeze(0).permute(1, 2, 0)

    return image


def screen_space_to_world_coords(intrinsic, extrinsics, image_coord, depth):
    """
    Convert points from screen space (image space) to world space.

    Args:
    intrinsic (torch.Tensor): The intrinsic matrix of the camera, shape (3, 3).
    extrinsics (torch.Tensor): The extrinsic matrix of the camera, shape (4, 4).
    screen_space (torch.Tensor): The screen space coordinates, shape (N, 2).
    depth (torch.Tensor): The depth values corresponding to the screen space coordinates, shape (N, 1).

    Returns:
    torch.Tensor: The world space coordinates, shape (N, 3).
    """
    batch = image_coord.shape[0]

    K_inv = torch.inverse(intrinsic).to(torch.float32)
    E = (extrinsics).to(torch.float32)
    R_inv = torch.inverse(extrinsics[:3, :3]).to(torch.float32)
    t = extrinsics[:3, 3].unsqueeze(0)

    screen_space = image_coord.clone().to(torch.float32)

    screen_space *= torch.tensor([1080.0, 1920.0]).to(torch.float32)

    screen_space[:, 0] = (screen_space[:, 0] - 1080/2)
    screen_space[:, 1] = (screen_space[:, 1] - 1920/2)

    screen_space[:, 0] = screen_space[:, 0] / intrinsic[0, 0]
    screen_space[:, 1] = screen_space[:, 1] / intrinsic[1, 1]

    screen_space = torch.cat((screen_space, torch.ones(
        (batch, 1)).to(torch.float32)), dim=1)  # Shape: (N, 4)

    camera_space = torch.einsum(
        'ij,bj->bi', K_inv, screen_space)  # Shape: (N, 3)

    camera_space *= depth

    world = torch.einsum('ij,bj->bi', R_inv, camera_space - t)  # Shape: (N, 3)

    return world


# def forward(intrinsic, extrinsics, world_space, camera):

#     batch = world_space.shape[0]

#     ones = torch.ones((batch, 1)).to(torch.float32)

#     # screen_space = torch.cat((screen_space, ones), dim=1)

#     world_space = world_space.to(torch.float32)

#     K = (intrinsic).to(torch.float32)
#     E = (extrinsics).to(torch.float32)
#     R = (extrinsics[:3, :3]).to(torch.float32)
#     t = extrinsics[:3, 3].unsqueeze(0)

#     camera_space = torch.einsum(
#         'ij,bj->bi', R, world_space)  # Shape: (N, 3)
#     camera_space = camera_space + t

#     image_space = torch.zeros_like(camera_space)
#     image_space[:, 0] = camera_space[:, 0] / camera_space[:, 2]
#     image_space[:, 1] = camera_space[:, 1] / camera_space[:, 2]
#     image_space[:, 2] = torch.sqrt(
#         camera_space[:, 0]**2 + camera_space[:, 1]**2 + camera_space[:, 2]**2)

#     magic_number = 1.2

#     image_space[:, 0] *= camera.fx
#     image_space[:, 1] *= camera.fy

#     image_space[:, 0] += camera.width/2
#     image_space[:, 1] += camera.height/2

#     return image_space


def forward(intrinsic, extrinsics, world_space, camera):

    batch = world_space.shape[0]

    ones = torch.ones((batch, 1)).to(torch.float32)

    # screen_space = torch.cat((screen_space, ones), dim=1)

    world_space = world_space.to(torch.float32)

    K = (intrinsic).to(torch.float32)
    E = (extrinsics).to(torch.float32)
    R = (extrinsics[:3, :3]).to(torch.float32)
    t = extrinsics[:3, 3].unsqueeze(0)

    camera_space = torch.einsum(
        'ij,bj->bi', R, world_space)  # Shape: (N, 3)
    camera_space = camera_space + t

    image_space = torch.einsum(
        'ij,bj->bi', K, camera_space)  # Shape: (N, 3)

    # magic_number = 1.2

    image_space[:, 0] = image_space[:, 0]*camera.fx/image_space[:, 2]
    image_space[:, 1] = image_space[:, 1]*camera.fy/image_space[:, 2]

    image_space[:, 0] += camera.cx
    image_space[:, 1] += camera.cy

    image_space[:, 0] /= camera.width
    image_space[:, 1] /= camera.height

    return image_space


if __name__ == "__main__":
    frame = 0
    downsample = 1
    # device = "cuda"
    dataset = ColmapDataset("dataset/nerfstudio/poster",
                            downsample_factor=downsample)

    estimator = Estimator()

    distance = 5

    width, height = dataset.camera.width, dataset.camera.height

    w, h = int(dataset.camera.width /
               downsample), int(dataset.camera.height/downsample)

    down_w, down_h = int(dataset.camera.width /
                         distance), int(dataset.camera.height/distance)

    # ipdb.set_trace()
    x = (torch.arange(down_w)*distance+0.5*distance).repeat(down_h, 1)
    y = (torch.arange(down_h)*distance+0.5*distance).repeat(down_w, 1).t()

    # x = (torch.arange(down_w)).repeat(down_h, 1)
    # y = (torch.arange(down_h)).repeat(down_w, 1).t()

    coords = torch.stack((x, y), dim=2).to(torch.float32)
    coords = coords.reshape(-1, 2)
    print("coords", coords)

    # for img_id in tqdm(range(0, len(dataset.image_info))):
    for img_id in tqdm(range(0, 1)):
        frame = img_id

        ground_truth = dataset.images[frame]
        ground_truth = ground_truth.to(torch.float)/255

        raw_image = ground_truth.numpy()

        K = dataset.camera.K_tensor.to(torch.float32)
        E = dataset.image_info[frame].extrinsic().to(torch.float32)
        R = E[:3, :3]
        t = E[:3, 3]
        trans = dataset.image_info[frame]

        depth = estimator.estimate(raw_image).cpu()

        # depth = resize_transform(depth)
        # depth = torch.ones((h, w, 1))
        # depth = torch.zeros((h, w, 1))
        depth = resize_image(depth, down_w, down_h)
        # depth = (depth-depth.min())/(depth.max()-depth.min())

        depth = depth.reshape(-1, 1)
        depth *= 0.1

        depth = 1/(depth+0.01)

        # print("detph", depth)

        point = dict()

        pos_2d = coords
        pos = screen_space_to_world_coords(K, E, pos_2d, depth)
        rgb = resize_image(ground_truth, down_w, down_h).reshape(-1, 3)

        axis_pos = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(torch.float32)
        pos = torch.cat((pos, axis_pos), dim=0)

        axis_rgb = torch.tensor(
            [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(torch.float32)
        rgb = torch.cat((rgb, axis_rgb), dim=0)

        point["pos"] = pos
        point["rgb"] = rgb

        # splatter = Splatter(load_ckpt="ckpt3.pth", downsample=downsample)
        splatter = Splatter(init_points=point, downsample=downsample)
        splatter.set_camera(dataset.camera)

        with torch.no_grad():
            render_image = splatter.forward(trans)

        render_image = render_image[:, :, :3]

        Splatter.save_image("output.png", render_image)

        time.sleep(0.1)
