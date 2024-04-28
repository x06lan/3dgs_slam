import argparse
import os
import cv2
import time
from tqdm import tqdm
from estimator import Estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str, default="dataset/depth/data")
    parser.add_argument("--outdir", type=str, default="dataset/depth/result")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="do not apply colorful palette")

    args = parser.parse_args()

    estimator = Estimator()

    filenames = os.listdir(args.img_path)
    t0 = time.time()
    for filename in tqdm(filenames):
        raw_image = cv2.imread(os.path.join(args.img_path, filename))
        depth = estimator.estimate(raw_image)
        cv2.imwrite(os.path.join(args.outdir, filename[: filename.rfind(".")] + "_depth.png"), depth)
    total = time.time() - t0
    print(f"Total time: {total:.2f}s")
    print(f"FPS: {len(filenames) / total:.2f}")
