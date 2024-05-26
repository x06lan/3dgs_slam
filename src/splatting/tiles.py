import math
import torch
import torch.nn as nn

import gaussian

from einops import repeat


class Tiles:
    def __init__(self, width, height, focal_x, focal_y, device):
        self.width = width
        self.height = height
        self.padded_width = int(math.ceil(self.width/16)) * 16
        self.padded_height = int(math.ceil(self.height/16)) * 16
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.n_tile_x = self.padded_width // 16
        self.n_tile_y = self.padded_height // 16
        self.device = device

    def crop(self, image):
        # image: padded_height x padded_width x 3
        # output: height x width x 3
        top = int(self.padded_height - self.height)//2
        left = int(self.padded_width - self.width)//2
        return image[top:top+int(self.height), left:left+int(self.width), :]

    def create_tiles(self):
        self.tiles_left = torch.linspace(-self.padded_width/2,
                                         self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + 16
        self.tiles_top = torch.linspace(-self.padded_height/2,
                                        self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + 16
        self.tile_geo_length_x = 16 / self.focal_x
        self.tile_geo_length_y = 16 / self.focal_y
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
