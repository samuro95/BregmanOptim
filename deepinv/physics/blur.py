from torchvision.transforms.functional import rotate
import torchvision
import torch
import numpy as np
import torch.fft as fft
from torch import Tensor
from deepinv.physics.forward import LinearPhysics, DecomposablePhysics
from deepinv.physics.functional import (
    conv2d,
    conv_transpose2d,
    filter_fft_2d,
    product_convolution2d,
    product_convolution2d_adjoint,
    product_convolution2d_patches,
    product_convolution2d_adjoint_patches,
    get_psf_product_convolution2d,
    get_psf_product_convolution2d_patches,
    conv3d_fft,
    conv_transpose3d_fft,
)
from deepinv.physics.functional.product_convolution import compute_patch_info


class Downsampling(LinearPhysics):
    r"""
    Downsampling operator for super-resolution problems.

    It is defined as

    .. math::

        y = S (h*x)

    where :math:`h` is a low-pass filter and :math:`S` is a subsampling operator.

    :param torch.Tensor, str, NoneType filter: Downsampling filter. It can be ``'gaussian'``, ``'bilinear'`` or ``'bicubic'`` or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param tuple[int] img_size: size of the input image
    :param int factor: downsampling factor
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    |sep|

    :Examples:

        Downsampling operator with a gaussian filter:

        >>> from deepinv.physics import Downsampling
        >>> x = torch.zeros((1, 1, 32, 32)) # Define black image of size 32x32
        >>> x[:, :, 16, 16] = 1 # Define one white pixel in the middle
        >>> physics = Downsampling(filter = "gaussian", img_size=(1, 32, 32), factor=2)
        >>> y = physics(x)
        >>> y[:, :, 7:10, 7:10] # Display the center of the downsampled image
        tensor([[[[0.0146, 0.0241, 0.0146],
                  [0.0241, 0.0398, 0.0241],
                  [0.0146, 0.0241, 0.0146]]]])

    """

    def __init__(
        self,
        img_size,
        filter=None,
        factor=2,
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = factor
        assert isinstance(
            factor, int), "downsampling factor should be an integer"
        # assert len(img_size) == 3, "img_size should be a tuple of length 3, C x H x W"
        self.imsize = img_size
        self.padding = padding
        if isinstance(filter, torch.nn.Parameter):
            self.filter = filter.requires_grad_(False).to(device)
        if isinstance(filter, torch.Tensor):
            self.filter = torch.nn.Parameter(
                filter, requires_grad=False).to(device)
        elif filter is None:
            self.filter = filter
        elif filter == "gaussian":
            self.filter = torch.nn.Parameter(
                gaussian_blur(sigma=(factor, factor)), requires_grad=False
            ).to(device)
        elif filter == "bilinear":
            self.filter = torch.nn.Parameter(
                bilinear_filter(self.factor), requires_grad=False
            ).to(device)
        elif filter == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(self.factor), requires_grad=False
            ).to(device)
        else:
            raise Exception("The chosen downsampling filter doesn't exist")

        if self.filter is not None:
            self.Fh = filter_fft_2d(
                self.filter, img_size, real_fft=False).to(device)
            self.Fhc = torch.conj(self.Fh)
            self.Fh2 = self.Fhc * self.Fh
            self.Fhc = torch.nn.Parameter(self.Fhc, requires_grad=False)
            self.Fh2 = torch.nn.Parameter(self.Fh2, requires_grad=False)

    def A(self, x, filter=None, **kwargs):
        r"""
        Applies the downsampling operator to the input image.

        :param torch.Tensor x: input image.
        :param None, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        if self.filter is not None:
            x = conv2d(x, self.filter, padding=self.padding)

        x = x[:, :, :: self.factor, :: self.factor]  # downsample
        return x

    def A_adjoint(self, y, filter=None, **kwargs):
        r"""
        Adjoint operator of the downsampling operator.

        :param torch.Tensor y: downsampled image.
        :param None, torch.Tensor filter: Filter :math:`h` to be applied to the input image before downsampling.
            If not ``None``, it uses this filter and stores it as the current filter.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(filter, requires_grad=False)

        imsize = self.imsize

        if self.filter is not None:
            if self.padding == "valid":
                imsize = (
                    self.imsize[0],
                    self.imsize[1] - self.filter.shape[-2] + 1,
                    self.imsize[2] - self.filter.shape[-1] + 1,
                )
            else:
                imsize = (
                    self.imsize[0],
                    self.imsize[1],
                    self.imsize[2],
                )

        x = torch.zeros((y.shape[0],) + imsize, device=y.device, dtype=y.dtype)
        x[:, :, :: self.factor, :: self.factor] = y  # upsample
        if self.filter is not None:
            x = conv_transpose2d(x, self.filter, padding=self.padding)
        return x

    def prox_l2(self, z, y, gamma, use_fft=True):
        r"""
        If the padding is circular, it computes the proximal operator with the closed-formula of
        https://arxiv.org/abs/1510.00143.

        Otherwise, it computes it using the conjugate gradient algorithm which can be slow if applied many times.
        """

        if use_fft and self.padding == "circular":  # Formula from (Zhao, 2016)
            z_hat = self.A_adjoint(y) + 1 / gamma * z
            Fz_hat = fft.fft2(z_hat)

            def splits(a, sf):
                """split a into sfxsf distinct blocks
                Args:
                    a: NxCxWxH
                    sf: split factor
                Returns:
                    b: NxCx(W/sf)x(H/sf)x(sf^2)
                """
                b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
                b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
                return b

            top = torch.mean(splits(self.Fh * Fz_hat, self.factor), dim=-1)
            below = torch.mean(
                splits(self.Fh2, self.factor), dim=-1) + 1 / gamma
            rc = self.Fhc * (top / below).repeat(1, 1,
                                                 self.factor, self.factor)
            r = torch.real(fft.ifft2(rc))
            return (z_hat - r) * gamma
        else:
            return LinearPhysics.prox_l2(self, z, y, gamma)


class Blur(LinearPhysics):
    r"""

    Blur operator.

    This forward operator performs

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    :param torch.Tensor filter: Tensor of size (b, 1, h, w) or (b, c, h, w) in 2D; (b, 1, d, h, w) or (b, c, d, h, w) in 3D, containing the blur filter, e.g., :meth:`deepinv.physics.blur.gaussian_filter`.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``. If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image. (default is ``'valid'``). Only ``padding='valid'`` and  ``padding = 'circular'`` are implemented in 3D.
    :param str device: cpu or cuda.


    .. note::

        This class makes it possible to change the filter at runtime by passing a new filter to the forward method, e.g.,
        ``y = physics(x, w)``. The new filter :math:`w` is stored as the current filter.

    .. note::

        This class uses the highly optimized :meth:`torch.nn.functional.conv2d` for performing the convolutions in 2D
        and FFT for performing the convolutions in 3D as implemented in :meth:`deepinv.physics.functional.conv3d_fft`.
        It uses FFT based convolutions in 3D since :meth:`torch.functional.nn.conv3d` is slow for large kernels.

    |sep|

    :Examples:

        Blur operator with a basic averaging filter applied to a 16x16 black image with
        a single white pixel in the center:

        >>> from deepinv.physics import Blur
        >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
        >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
        >>> w = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 averaging filter
        >>> physics = Blur(filter=w)
        >>> y = physics(x)
        >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
        tensor([[[[0.2500, 0.2500, 0.0000],
                  [0.2500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])

    """

    def __init__(self, filter=None, padding="valid", device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.padding = padding
        self.update_parameters(filter)

    def A(self, x, filter=None, **kwargs):
        r"""
        Applies the filter to the input image.

        :param torch.Tensor x: input image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """
        self.update_parameters(filter)

        if x.dim() == 4:
            return conv2d(x, filter=self.filter, padding=self.padding)
        elif x.dim() == 5:
            return conv3d_fft(x, filter=self.filter, padding=self.padding)

    def A_adjoint(self, y, filter=None, **kwargs):
        r"""
        Adjoint operator of the blur operator.

        :param torch.Tensor y: blurred image.
        :param torch.Tensor filter: Filter :math:`w` to be applied to the input image.
            If not ``None``, it uses this filter instead of the one defined in the class, and
            the provided filter is stored as the current filter.
        """
        self.update_parameters(filter)

        if y.dim() == 4:
            return conv_transpose2d(y, filter=self.filter, padding=self.padding)
        elif y.dim() == 5:
            return conv_transpose3d_fft(y, filter=self.filter, padding=self.padding)

    def update_parameters(self, filter=None, **kwargs):
        r"""
        Updates the current filter.

        :param torch.Tensor filter: New filter to be applied to the input image.
        """
        if filter is not None:
            self.filter = torch.nn.Parameter(
                filter.to(self.device), requires_grad=False
            )

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)


class BlurFFT(DecomposablePhysics):
    """

    FFT-based blur operator.

    It performs the operation

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    Blur operator based on ``torch.fft`` operations, which assumes a circular padding of the input, and allows for
    the singular value decomposition via ``deepinv.Physics.DecomposablePhysics`` and has fast pseudo-inverse and prox operators.



    :param tuple img_size: Input image size in the form (C, H, W).
    :param torch.Tensor filter: torch.Tensor of size (1, c, h, w) containing the blur filter with h<=H, w<=W and c=1 or c=C e.g.,
        :meth:`deepinv.physics.blur.gaussian_filter`.
    :param str device: cpu or cuda

    |sep|

    :Examples:

        BlurFFT operator with a basic averaging filter applied to a 16x16 black image with
        a single white pixel in the center:

        >>> from deepinv.physics import BlurFFT
        >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
        >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
        >>> filter = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 filter
        >>> physics = BlurFFT(filter=filter, img_size=(1, 1, 16, 16))
        >>> y = physics(x)
        >>> y[y<1e-5] = 0.
        >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
        tensor([[[[0.2500, 0.2500, 0.0000],
                  [0.2500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(self, img_size, filter=None, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.img_size = img_size
        self.update_parameters(filter=filter, **kwargs)

    def A(self, x, filter=None, **kwargs):
        self.update_parameters(filter)
        return super().A(x)

    def A_adjoint(self, x, filter=None, **kwargs):
        self.update_parameters(filter)
        return super().A_adjoint(x)

    def V_adjoint(self, x):
        return torch.view_as_real(
            fft.rfft2(x, norm="ortho")
        )  # make it a true SVD (see J. Romberg notes)

    def U(self, x):
        return fft.irfft2(
            torch.view_as_complex(x) * self.angle,
            norm="ortho",
            s=self.img_size[-2:],
        )

    def U_adjoint(self, x):
        return torch.view_as_real(
            fft.rfft2(x, norm="ortho") * torch.conj(self.angle)
        )  # make it a true SVD (see J. Romberg notes)

    def V(self, x):
        return fft.irfft2(torch.view_as_complex(x), norm="ortho", s=self.img_size[-2:])

    def update_parameters(self, filter=None, **kwargs):
        r"""
        Updates the current filter.

        :param torch.Tensor filter: New filter to be applied to the input image.
        """
        if filter is not None:
            if self.img_size[0] > filter.shape[1]:
                filter = filter.repeat(1, self.img_size[0], 1, 1)
            self.filter = torch.nn.Parameter(filter, requires_grad=False).to(
                self.device
            )

            mask = filter_fft_2d(filter, self.img_size).to(self.device)
            self.angle = torch.angle(mask)
            self.angle = torch.exp(-1.0j * self.angle).to(self.device)
            mask = torch.abs(mask).unsqueeze(-1)
            mask = torch.cat([mask, mask], dim=-1)
            self.mask = torch.nn.Parameter(mask, requires_grad=False)

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)


class SpaceVaryingBlur(LinearPhysics):
    r"""

    Implements a space varying blur via product-convolution.

    This operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product,  :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W.
    :param str method: 'product_convolution' or 'product_convolution2d_patch'.
    :param patch_info: dictionary of patch information: patch_size, overlap, number of patches in each dimension
    :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        If ``padding = 'valid'`` the blurred output is smaller than the image (no padding),
        otherwise the blurred output has the same size as the image.

    :param str device: cpu or cuda

    |sep|
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    :Examples:

        We show how to instantiate a spatially varying blur operator.

        >>> from deepinv.physics.generator import DiffractionBlurGenerator, ProductConvolutionBlurGenerator
        >>> from deepinv.physics.blur import SpaceVaryingBlur
        >>> from deepinv.utils.plotting import plot
        >>> psf_size = 32
        >>> img_size = (256, 256)
        >>> delta = 16
        >>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size))
        >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator=psf_generator, img_size=img_size)
        >>> params_pc = pc_generator.step(1)
        >>> physics = SpaceVaryingBlur(**params_pc)
        >>> dirac_comb = torch.zeros(img_size).unsqueeze(0).unsqueeze(0)
        >>> dirac_comb[0,0,::delta,::delta] = 1
        >>> psf_grid = physics(dirac_comb)
        >>> plot(psf_grid, titles="Space varying impulse responses")

    """

    def __init__(self, filters=None, multipliers=None, padding=None, method: str = 'product_convolution2d', patch_size: Tuple[int] = None, overlap: Tuple[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        if method == 'product_convolution2d_patch':
            self.patch_size = patch_size
            self.overlap = overlap

        self.update_parameters(filters, multipliers, padding)
        self.image_size = None

    def A(
        self, x: Tensor, filters=None, multipliers=None, padding=None, patch_size: Tuple[int] = None, overlap: Tuple[int] = None, ** kwargs
    ) -> Tensor:
        r"""
        Applies the space varying blur operator to the input image.

        It can receive new parameters  :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor filters: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
        :param torch.Tensor multipliers: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        self.update_parameters(filters, multipliers, padding)
        if self.method == "product_convolution2d":
            return product_convolution2d(
                x, self.multipliers, self.filters, self.padding
            )
        elif self.method == "product_convolution2d_patch":
            if patch_size is not None:
                self.patch_size = patch_size
            if overlap is not None:
                self.overlap = overlap

            if tuple(x.shape[-2:]) != self.image_size:
                self.image_size = tuple(x.shape[-2:])
                self.update_patch_info(
                    self.image_size, self.patch_size, self.overlap)
            self.check_patch_info()
            return product_convolution2d_patches(x,
                                                 w=self.multipliers,
                                                 h=self.filters,
                                                 patch_size=self.patch_size,
                                                 overlap=self.overlap)
        else:
            raise NotImplementedError(
                "Method not implemented in product-convolution")

    def A_adjoint(
        self, y: Tensor, filters=None, multipliers=None, padding=None, patch_size: Tuple[int] = None, overlap: Tuple[int] = None, **kwargs
    ) -> Tensor:
        r"""
        Applies the adjoint operator.

        It can receive new parameters :math:`w_k`, :math:`h_k` and padding to be used in the forward operator, and stored
        as the current parameters.

        :param torch.Tensor h: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param torch.Tensor w: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
            If `padding = 'valid'` the blurred output is smaller than the image (no padding),
            otherwise the blurred output has the same size as the image.
        :param str device: cpu or cuda
        """
        if self.method == "product_convolution2d":
            self.update_parameters(filters, multipliers, padding)

            return product_convolution2d_adjoint(
                y, self.multipliers, self.filters, self.padding
            )
        elif self.method == "product_convolution2d_patch":
            if patch_size is not None:
                self.patch_size = patch_size
            if overlap is not None:
                self.overlap = overlap

            self.update_patch_info(
                self.image_size, self.patch_size, self.overlap)
            self.check_patch_info()

            return product_convolution2d_adjoint_patches(y, w=self.multipliers, h=self.filters, patch_size=self.patch_size, overlap=self.overlap)

        else:
            raise NotImplementedError(
                "Method not implemented in product-convolution")

    def get_psf(self, centers: Tensor = None, patch_size: Tuple[int] = None, overlap: Tuple[int] = None, **kwargs):
        r"""
        :param torch.Tensor centers: (B, num_center_per_batch, 2)

        :return: (num_patch_psf, B, C, psf_size, psf_size)
        """
        self.update_parameters(**kwargs)
        h = self.filters
        w = self.multipliers

        if self.method == "product_convolution2d_patch":
            if patch_size is not None:
                self.patch_size = patch_size
            if overlap is not None:
                self.overlap = overlap

            self.update_patch_info(
                self.image_size, self.patch_size, self.overlap)
            self.check_patch_info()

        # Method 1: Simple nested loops
        psf = []
        for b in range(centers.size(0)):
            for k in range(centers.size(1)):
                position = centers[b, k, :]
                if self.method == 'product_convolution2d':
                    psf.append(get_psf_product_convolution2d(
                        h[:, b:b+1, ...], w[:, b:b+1, ...], position))
                elif self.method == 'product_convolution2d_patch':
                    psf.append(get_psf_product_convolution2d_patches(
                        h[:, b:b+1, ...], w, position, overlap=self.overlap, num_patches=self.num_patches))
        return torch.stack(psf, dim=0)

    def get_psf_2(self, centers: Tensor = None, patch_size: Tuple[int] = None, overlap: Tuple[int] = None, **kwargs):
        r"""
        :param torch.Tensor centers: (B, num_center_per_batch, 2)

        :return: (B, num_center_per_batch, C, psf_size, psf_size)
        """
        self.update_parameters(**kwargs)
        h = self.filters
        w = self.multipliers

        if self.method == "product_convolution2d_patch":
            if patch_size is not None:
                self.patch_size = patch_size
            if overlap is not None:
                self.overlap = overlap

            self.update_patch_info(
                self.image_size, self.patch_size, self.overlap)
            self.check_patch_info()

        psf = []
        if self.method == 'product_convolution2d_patch':
            for k in range(centers.size(1)):
                position = centers[:, k, :]
                psf.append(get_psf_product_convolution2d_patches_v2(
                    h, w, position, overlap=self.overlap, num_patches=self.num_patches))
        else:
            raise ValueError(f'Unsupported method {self.method}')
        return torch.stack(psf, dim=1)

    def get_psf_3(self, centers: Tensor = None, patch_size: Tuple[int] = None, overlap: Tuple[int] = None, **kwargs):
        r"""
        :param torch.Tensor centers: (B, num_center_per_batch, 2)

        :return: (B, num_center_per_batch, C, psf_size, psf_size)
        """
        self.update_parameters(**kwargs)
        h = self.filters
        w = self.multipliers

        if self.method == "product_convolution2d_patch":
            if patch_size is not None:
                self.patch_size = patch_size
            if overlap is not None:
                self.overlap = overlap

            self.update_patch_info(
                self.image_size, self.patch_size, self.overlap)
            self.check_patch_info()

        psf = []
        if self.method == 'product_convolution2d_patch':
            psf = get_psf_product_convolution2d_patches_v3(
                h, w, centers, overlap=self.overlap, num_patches=self.num_patches)
        else:
            raise ValueError(f'Unsupported method {self.method}')
        return psf

    def update_parameters(self, filters=None, multipliers=None, padding=None, **kwargs):
        r"""
        Updates the current parameters.

        :param torch.Tensor filters: Multipliers :math:`w_k`. Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
        :param torch.Tensor multipliers: Filters :math:`h_k`. Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
        :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        """
        if filters is not None:
            self.filters = torch.nn.Parameter(filters, requires_grad=False)
        if multipliers is not None:
            self.multipliers = torch.nn.Parameter(
                multipliers, requires_grad=False)
        if padding is not None:
            self.padding = padding

    def update_patch_info(self, image_size, patch_size, overlap):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(overlap, int):
            overlap = (overlap, overlap)

        self.image_size = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.num_patches = compute_patch_info(
            image_size, patch_size, overlap)['num_patches']

    def check_patch_info(self):
        if self.patch_size is None or self.overlap is None:
            raise ValueError(
                "Patch information is required for product_convolution2d_patch method. Please specify the patch_size and overlap")


def gaussian_blur(sigma=(1, 1), angle=0):
    r"""
    Gaussian blur filter.

    Defined as

    .. math::
        \begin{equation*}
            G(x, y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp{\left(-\frac{x'^2}{2\sigma_x^2} - \frac{y'^2}{2\sigma_y^2}\right)}
        \end{equation*}

    where :math:`x'` and :math:`y'` are the rotated coordinates obtained by rotating $(x, y)$ around the origin
    by an angle :math:`\theta`:

    .. math::

        \begin{align*}
            x' &= x \cos(\theta) - y \sin(\theta) \\
            y' &= x \sin(\theta) + y \cos(\theta)
        \end{align*}

    with :math:`\sigma_x` and :math:`\sigma_y`  the standard deviations along the :math:`x'` and :math:`y'` axes.


    :param float, tuple[float] sigma: standard deviation of the gaussian filter. If sigma is a float the filter is isotropic, whereas
        if sigma is a tuple of floats (sigma_x, sigma_y) the filter is anisotropic.
    :param float angle: rotation angle of the filter in degrees (only useful for anisotropic filters)
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    s = max(sigma)
    c = int(s / 0.3 + 1)
    k_size = 2 * c + 1

    delta = torch.arange(k_size)

    x, y = torch.meshgrid(delta, delta, indexing="ij")
    x = x - c
    y = y - c
    filt = (x / sigma[0]).pow(2)
    filt += (y / sigma[1]).pow(2)
    filt = torch.exp(-filt / 2.0)

    filt = (
        rotate(
            filt.unsqueeze(0).unsqueeze(0),
            angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        .squeeze(0)
        .squeeze(0)
    )

    filt = filt / filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)


def kaiser_window(beta, length):
    """Return the Kaiser window of length `length` and shape parameter `beta`."""
    if beta < 0:
        raise ValueError("beta must be greater than 0")
    if length < 1:
        raise ValueError("length must be greater than 0")
    if length == 1:
        return torch.tensor([1.0])
    half = (length - 1) / 2
    n = torch.arange(length)
    beta = torch.tensor(beta)
    return torch.i0(beta * torch.sqrt(1 - ((n - half) / half) ** 2)) / torch.i0(beta)


def sinc_filter(factor=2, length=11, windowed=True):
    r"""
    Anti-aliasing sinc filter multiplied by a Kaiser window.

    The kaiser window parameter is computed as follows:

    .. math::

        A = 2.285 \cdot (L - 1) \cdot 3.14 \cdot \Delta f + 7.95

    where :math:`\Delta f = 1 / \text{factor}`. Then, the beta parameter is computed as:

    .. math::

        \begin{equation*}
            \beta = \begin{cases}
                0 & \text{if } A \leq 21 \\
                0.5842 \cdot (A - 21)^{0.4} + 0.07886 \cdot (A - 21) & \text{if } 21 < A \leq 50 \\
                0.1102 \cdot (A - 8.7) & \text{otherwise}
            \end{cases}
        \end{equation*}

    :param float factor: Downsampling factor.
    :param int length: Length of the filter.
    """
    deltaf = 1 / factor

    n = torch.arange(length) - (length - 1) / 2
    filter = torch.sinc(n / factor)

    if windowed:
        A = 2.285 * (length - 1) * 3.14 * deltaf + 7.95
        if A <= 21:
            beta = 0
        elif A <= 50:
            beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.1102 * (A - 8.7)

        filter = filter * kaiser_window(beta, length)

    filter = filter.unsqueeze(0)
    filter = filter * filter.T
    filter = filter.unsqueeze(0).unsqueeze(0)
    filter = filter / filter.sum()
    return filter


def bilinear_filter(factor=2):
    r"""
    Bilinear filter.

    It has size (2*factor, 2*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (1 - |x|) \cdot (1 - |y|) & \text{if } |x| \leq 1 \text{ and } |y| \leq 1 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-\text{factor} + 0.5, -\text{factor} + 0.5 + 1/\text{factor}, \ldots, \text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-factor + 0.5, stop=factor, step=1) / factor
    w = 1 - np.abs(x)
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)
