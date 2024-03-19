import torch
from deepinv.physics.noise import GaussianNoise
from deepinv.physics.forward import LinearPhysics
from deepinv.physics.blur.blur import Downsampling
from deepinv.physics.range import Decolorize
from deepinv.utils import TensorList


class Pansharpen(LinearPhysics):
    r"""
    Pansharpening forward operator.

    The measurements consist of a high resolution grayscale image and a low resolution RGB image, and
    are represented using :class:`deepinv.utils.TensorList`, where the first element is the RGB image and the second
    element is the grayscale image.

    By default, the downsampling is done with a gaussian filter with standard deviation equal to the downsampling,
    however, the user can provide a custom downsampling filter.

    It is possible to assign a different noise model to the RGB and grayscale images.

    :param tuple[int] img_size: size of the input image.
    :param int factor: downsampling factor.
    :param torch.nn.Module noise_color: noise model for the RGB image.
    :param torch.nn.Module noise_gray: noise model for the grayscale image.
    :param torch.Tensor, str, NoneType filter: Downsampling filter. It can be 'gaussian', 'bilinear' or 'bicubic' or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    |sep|

    :Examples:

        Pansharpen operator applied to a random 32x32 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 3, 32, 32) # Define random 32x32 image
        >>> physics = Pansharpen(img_size=x.shape[1:], device=x.device)
        >>> physics(x)[0][:, :, 0, :3] # Display first pixels of RGB image
        tensor([[[-0.0009, -0.0251, -0.0411],
                 [-0.1576, -0.1098, -0.0340],
                 [ 0.0086, -0.0257, -0.0856]]])
        >>> physics(x)[1][:, :, 0, :3] # Display first pixels of grayscale image
        tensor([[[-0.9084, -0.2966, -0.4015]]])

    """

    def __init__(
        self,
        img_size,
        params,
        factor=4,
        noise_color=GaussianNoise(sigma=0.0),
        noise_gray=GaussianNoise(sigma=0.05),
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downsampling = Downsampling(
            img_size=img_size,
            factor=factor,
            params=params,
            device=device,
            padding=padding,
        )

        self.noise_color = noise_color
        self.noise_gray = noise_gray
        self.colorize = Decolorize()

    def A(self, x):
        return TensorList([self.downsampling.A(x), self.colorize.A(x)])

    def A_adjoint(self, y):
        return self.downsampling.A_adjoint(y[0]) + self.colorize.A_adjoint(y[1])

    def forward(self, x):
        return TensorList(
            [self.noise_color(self.downsampling(x)), self.noise_gray(self.colorize(x))]
        )


# test code
# if __name__ == "__main__":
#     device = "cuda:0"
#     import torch
#     import torchvision
#     import deepinv
#
#     device = "cuda:0"
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(device) / 255
#     x = torchvision.transforms.Resize((160, 180))(x)
#
#     class Toy(LinearPhysics):
#         def __init__(self, **kwargs):
#             super().__init__(**kwargs)
#             self.A = lambda x: x * 2
#             self.A_adjoint = lambda x: x * 2
#
#     sigma_noise = 0.1
#     kernel = torch.zeros((1, 1, 15, 15), device=device)
#     kernel[:, :, 7, :] = 1 / 15
#     # physics = deepinv.physics.BlurFFT(img_size=x.shape[1:], filter=kernel, device=device)
#     physics = Pansharpen(factor=8, img_size=x.shape[1:], device=device)
#
#     y = physics(x)
#
#     xhat2 = physics.A_adjoint(y)
#     xhat1 = physics.A_dagger(y)
#
#     physics.compute_norm(x)
#     physics.adjointness_test(x)
#
#     deepinv.utils.plot(
#         [y[0], y[1], xhat2, xhat1, x],
#         titles=["low res color", "high res gray", "A_adjoint", "A_dagger", "x"],
#     )
