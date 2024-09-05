import torch
from torchvision.transforms.functional import rotate
import numpy as np
from deepinv.transform.base import Transform
import torch.fft


class Rotate(Transform):
    r"""
    2D Rotations.

    Generates n_transf randomly rotated versions of 2D images with zero padding.

    :param degrees: images are rotated in the range of angles (-degrees, degrees)
    :param n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, degrees=360, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = degrees

    def forward(self, x):
        r"""
        Applies a random rotation to the input image.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :return: torch.Tensor containing the rotated images concatenated along the first dimension
        """
        if self.group_size == 360:
            theta = np.arange(0, 360)[1:][torch.randperm(359, generator=self.rng)]
            theta = theta[: self.n_trans]
        else:
            theta = np.arange(0, 360, int(360 / (self.group_size + 1)))[1:]
            theta = theta[torch.randperm(self.group_size, generator=self.rng)][
                : self.n_trans
            ]
        return torch.cat([rotate(x, float(_theta)) for _theta in theta])


def rotate_image_via_shear(image: torch.Tensor, angle_deg: torch.Tensor, center=None):
    r"""
    2D rotation of image by angle via shear composition through FFT.
    
    :param torch.Tensor image: input image of shape (B,C,H,W)
    :param torch.Tensor angle_deg: input rotation angles in degrees of shape (B,)
    :return: torch.Tensor containing the rotated images of shape (B, C, H, W )
    """
    # Convert angle to radians
    angle = torch.deg2rad(angle_deg)
    N0, N1 = image.shape[-2], image.shape[-1]
    if center is None:
        center = (N0//2, N1//2)
  
    mask_angles = (angle > torch.pi / 2.0) & (angle <=  3 * torch.pi / 2)

    angle[angle > 3 * torch.pi / 2] -= 2 * torch.pi 
    
    transformed_image = torch.zeros_like(image).expand(mask_angles.shape[0], -1, -1, -1).clone()
    expanded_image = image.clone().expand(mask_angles.shape[0], -1, -1, -1).clone()
    transformed_image[~mask_angles] = expanded_image[~mask_angles]
    transformed_image[mask_angles] = torch.rot90(expanded_image[mask_angles], k=-2, dims=(-2, -1))
    
    angle[mask_angles] -= torch.pi

    tant2 = - torch.tan(-angle/ 2)
    st = torch.sin(-angle)

    def shearx(image, shear):
        fft1 = torch.fft.fft2(image, dim=(-1))
        freq_1 = torch.fft.fftfreq(N1, d=1.0, device=image.device)
        freq_0 = shear[:, None] * (torch.arange(N0, device=image.device) - center[0])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[..., None] * freq_1[None, None, :])
        image_shear = fft1 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-1)))
    
    def sheary(image, shear):
        fft0 = torch.fft.fft2(image, dim=(-2))
        freq_0 = torch.fft.fftfreq(N0, d=1.0, device=image.device)
        freq_1 = shear[:, None] * (torch.arange(N1, device=image.device) - center[1])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[None, :, None] * freq_1[:, None, :])
        image_shear = fft0 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-2)))
        
    rot = shearx(sheary(shearx(transformed_image, tant2), st), tant2)
    return rot 


# if __name__ == "__main__":
#     device = "cuda:0"
#
#     x = torch.zeros(1, 1, 64, 64, device=device)
#     x[:, :, 16:48, 16:48] = 1
#
#     t = Rotate(4)
#     y = t(x)
#
#     from deepinv.utils import plot
#
#     plot([x, y[0, :, :, :].unsqueeze(0), y[1, :, :, :].unsqueeze(0)])
