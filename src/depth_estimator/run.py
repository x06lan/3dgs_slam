import argparse
import torch
import cv2
import numpy as np
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--img-path", type=str, default="dataset/depth/data")
    parser.add_argument("--outdir", type=str, default="dataset/depth/result")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="do not apply colorful palette")

    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    }

    depth_anything = DepthAnything(model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f"src/depth_estimator/depth_anything/checkpoints/depth_anything_{args.encoder}14.pth"))
    if DEVICE == "cuda":
        depth_anything.cuda()

    # must be multiple of 14
    img_dim = 518

    transform = Compose(
        [
            Resize(
                width=img_dim,
                height=img_dim,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    filenames = os.listdir(args.img_path)
    t0 = time.time()
    for filename in tqdm(filenames):
        raw_image = cv2.imread(os.path.join(args.img_path, filename))
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.outdir, filename[: filename.rfind(".")] + "_depth.png"), depth)
    total = time.time() - t0
    print(f"Total time: {total:.2f}s")
    print(f"FPS: {len(filenames) / total:.2f}")
