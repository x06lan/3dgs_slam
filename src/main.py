import torch

# import gaussian_cuda

from tracker import visual_odometry
from parser.dataset import ColampDataset

if __name__ == "__main__":

    dataset = ColampDataset(".//dataset/nerfstudio/person")
    pass
