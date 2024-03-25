import numpy as np
import torch
from deepinv.physics.generator import Generator


class AccelerationMaskGenerator(Generator):
    r"""
    Generator for MRI cartesian acceleration masks.

    It generates a mask of vertical lines for MRI acceleration using fixed sampling in the low frequencies (center of k-space),
    and random uniform sampling in the high frequencies.

    :param tuple image_size: image size.
    :param int acceleration: acceleration factor.
    :param str device: cpu or gpu.
    """

    def __init__(self, image_size: tuple, acceleration=4, device: str = "cpu"):
        super().__init__(shape=image_size, device=device)
        self.device = device
        self.image_size = image_size
        self.acceleration = acceleration

    def sample_mask(self, image_size, acceleration_factor=4, seed=None):
        r"""
        Create a mask of vertical lines.

        :param tuple image_size: image size.
        :param int acceleration_factor: acceleration factor.
        :param int seed: random seed.
        :return: mask of size (H, W) with values in {0, 1}.
        """
        if seed is not None:
            np.random.seed(seed)
        if acceleration_factor == 4:
            central_lines_percent = 0.08
            num_lines_center = int(central_lines_percent * image_size[-1])
            side_lines_percent = 0.25 - central_lines_percent
            num_lines_side = int(side_lines_percent * image_size[-1])
        if acceleration_factor == 8:
            central_lines_percent = 0.04
            num_lines_center = int(central_lines_percent * image_size[-1])
            side_lines_percent = 0.125 - central_lines_percent
            num_lines_side = int(side_lines_percent * image_size[-1])
        mask = torch.zeros(image_size)
        center_line_indices = torch.linspace(
            image_size[0] // 2 - num_lines_center // 2,
            image_size[0] // 2 + num_lines_center // 2 + 1,
            steps=50,
            dtype=torch.long,
        )
        mask[:, center_line_indices] = 1
        random_line_indices = np.random.choice(
            image_size[0], size=(num_lines_side // 2,), replace=False
        )
        mask[:, random_line_indices] = 1
        return mask.float().to(self.device)
