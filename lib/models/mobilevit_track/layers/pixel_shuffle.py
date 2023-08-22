#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Tuple


class PixelShuffle(nn.PixelShuffle):
    """
    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C \times r^2, H, W)`, where * is zero or more dimensions
        - Output: :math:`(*, C, H \times r, W \times r)`
    """

    def __init__(self, upscale_factor: int, *args, **kwargs) -> None:
        super(PixelShuffle, self).__init__(upscale_factor=upscale_factor)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return "{}(upscale_factor={})".format(
            self.__class__.__name__, self.upscale_factor
        )
