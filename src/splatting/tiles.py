import math
import torch
import torch.nn as nn

import gaussian

from einops import repeat


class Tiles:
    def __init__(self, width, height, focal_x, focal_y, device):
        self.width: int = width
        self.height: int = height
        self.tile_size: int = 16
        self.padded_width: int = int(
            math.ceil(self.width/self.tile_size)) * self.tile_size
        self.padded_height: int = int(
            math.ceil(self.height/self.tile_size)) * self.tile_size
        self.focal_x: float = focal_x
        self.focal_y = focal_y
        self.n_tile_x: float = self.padded_width // self.tile_size
        self.n_tile_y: float = self.padded_height // self.tile_size
        self.device = device

    def __len__(self):
        return self.tiles_top.shape[0]

    def crop(self, image):
        # image: padded_height x padded_width x 3
        # output: height x width x 3
        top = int(self.padded_height - self.height)//2
        left = int(self.padded_width - self.width)//2
        return image[top:top+int(self.height), left:left+int(self.width), :]

    def create_tiles(self):
        self.tiles_left = torch.linspace(-self.padded_width/2,
                                         self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + self.tile_size
        self.tiles_top = torch.linspace(-self.padded_height/2,
                                        self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + self.tile_size
        self.tile_geo_length_x = self.tile_size / self.focal_x
        self.tile_geo_length_y = self.tile_size / self.focal_y
        self.leftmost = -self.padded_width/2/self.focal_x
        self.topmost = -self.padded_height/2/self.focal_y

        self.tiles_left = self.tiles_left/self.focal_x
        self.tiles_top = self.tiles_top/self.focal_y
        self.tiles_right = self.tiles_right/self.focal_x
        self.tiles_bottom = self.tiles_bottom/self.focal_y

        self.tiles_left = repeat(
            self.tiles_left, "b -> (c b)", c=self.n_tile_y)
        self.tiles_right = repeat(
            self.tiles_right, "b -> (c b)", c=self.n_tile_y)

        self.tiles_top = repeat(self.tiles_top, "b -> (b c)", c=self.n_tile_x)
        self.tiles_bottom = repeat(
            self.tiles_bottom, "b -> (b c)", c=self.n_tile_x)

        _tile = gaussian.Tiles()
        _tile.top = self.tiles_top
        _tile.bottom = self.tiles_bottom
        _tile.left = self.tiles_left
        _tile.right = self.tiles_right
        return _tile
