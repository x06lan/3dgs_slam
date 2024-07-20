import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class Estimator:
    def __init__(self):
        # config
        self.conig = {
            "encoder": "vits",
            "input_dim": 518,
            "is_gray_scale": False,
        }

        # setup model
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model_configs = {
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        }
        self.model = DepthAnything(model_configs[self.conig["encoder"]])
        self.model.load_state_dict(torch.load(
            f"src/depth_estimator/depth_anything/checkpoints/depth_anything_{self.conig['encoder']}14.pth"))
        if self.DEVICE == "cuda":
            self.model.cuda()

        # transform
        self.transform = Compose(
            [
                Resize(
                    width=self.conig["input_dim"],
                    height=self.conig["input_dim"],
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def estimate(self, image):
        """
        Args:
            image (_type_): BGR image
        Returns:
            depth (_type_): depth map
        """
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        image = self.transform({"image": image})["image"]

        image = torch.from_numpy(image)

        image = image.unsqueeze(0).to(self.DEVICE)
        # image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            depth = self.model(image)

        # depth = F.interpolate(depth[None], (h, w),
        #                       mode="bilinear", align_corners=False)[0, 0]
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # depth = depth.cpu().numpy().astype(np.uint8)
        # if self.conig["is_gray_scale"]:
        #     depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        # else:
        #     depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return depth.permute(1, 2, 0)
