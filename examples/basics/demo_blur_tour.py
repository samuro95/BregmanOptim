r"""
A tour of blur operators
===================================================

This example provides a tour of 2D blur operators in DeepInv.
In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.

"""

# %%
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
import deepinv as dinv
import torch
from deepinv.physics.generator import MotionBlurGenerator
from deepinv.physics.generator import DiffractionBlurGenerator
from deepinv.physics.generator import GeneratorMixture

from deepinv.physics.generator import ProductConvolutionPatchBlurGenerator
from deepinv.physics.generator import (
    DiffractionBlurGenerator,
    ProductConvolutionBlurGenerator,
    ProductConvolutionPatchBlurGenerator,
)
from deepinv.physics.blur import SpaceVaryingBlur
from deepinv.physics.functional.product_convolution import compute_patch_info

# %%
# Load test images
# ----------------
#
# First, let's load some test images.

dtype = torch.float32
device = "cpu"
img_size = (173, 125)

url = get_image_url("CBSD_0010.png")
x_rgb = load_url_image(
    url, grayscale=False, device=device, dtype=dtype, img_size=img_size
)

url = get_image_url("barbara.jpeg")
x_gray = load_url_image(
    url, grayscale=True, device=device, dtype=dtype, img_size=img_size
)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %%
# We are now ready to explore the different blur operators.
#
# Convolution Basics
# ------------------
#
# The class :class:`deepinv.physics.Blur` implements convolution operations with kernels.
#
# For instance, here is the convolution of a grayscale image with a grayscale filter:
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device)
y = physics(x_gray)
plot(
    [x_gray, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Grayscale convolution",
)

# %%
# When a single channel filter is used, all channels are convolved with the same filter:
#

physics = dinv.physics.Blur(filter_0, device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="RGB image + grayscale filter convolution",
)

# %%
# By default, the boundary conditions are ``'valid'``, but other options among (``'circular'``, ``'reflect'``, ``'replicate'``) are possible:
#

physics = dinv.physics.Blur(filter_0, padding="reflect", device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Reflection boundary conditions",
)

# %%
# For circular boundary conditions, an FFT implementation is also available. It is slower that :meth:`deepinv.physics.Blur`,
# but inherits from :meth:`deepinv.physics.DecomposablePhysics`, so that the pseudo-inverse and regularized inverse are computed faster and more accurately.
#
physics = dinv.physics.BlurFFT(img_size=x_rgb[0].shape, filter=filter_0, device=device)
y = physics(x_rgb)
plot(
    [x_rgb, filter_0, y],
    titles=["signal", "filter", "measurement"],
    suptitle="FFT convolution with circular boundary conditions",
)

# %%
# One can also change the blur filter in the forward pass as follows:
filter_90 = dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=90.0).to(
    device=device, dtype=dtype
)
y = physics(x_rgb, filter=filter_90)
plot(
    [x_rgb, filter_90, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Changing the filter on the fly",
)

# %%
# When applied to a new image, the last filter is used:
y = physics(x_gray, filter=filter_90)
plot(
    [x_gray, filter_90, y],
    titles=["signal", "filter", "measurement"],
    suptitle="Effect of on the fly change is persistent",
)

# %%
# We can also define color filters. In that situation, each channel is convolved with the corresponding channel of the filter:
psf_size = 9
filter_rgb = torch.zeros((1, 3, psf_size, psf_size), device=device, dtype=dtype)
filter_rgb[:, 0, :, psf_size // 2 : psf_size // 2 + 1] = 1.0 / psf_size
filter_rgb[:, 1, psf_size // 2 : psf_size // 2 + 1, :] = 1.0 / psf_size
filter_rgb[:, 2, ...] = (
    torch.diag(torch.ones(psf_size, device=device, dtype=dtype)) / psf_size
)
y = physics(x_rgb, filter=filter_rgb)
plot(
    [x_rgb, filter_rgb, y],
    titles=["signal", "Colour filter", "measurement"],
    suptitle="Color image + color filter convolution",
)

# %%
# Blur generators
# ----------------------
# More advanced kernel generation methods are provided with the toolbox thanks to
# the  :class:`deepinv.physics.generator.PSFGenerator`. In particular, motion blurs generators are implemented.

# %%
# Motion blur generators
# ~~~~~~~~~~~~~~~~~~~~~~

# %%
# In order to generate motion blur kernels, we just need to instantiate a generator with specific the psf size.
# In turn, motion blurs can be generated on the fly by calling the ``step()`` method. Let's illustrate this now and
# generate 3 motion blurs. First, we instantiate the generator:
#
psf_size = 31
motion_generator = MotionBlurGenerator((psf_size, psf_size), device=device, dtype=dtype)
# %%
# To generate new filters, we call the step() function:
filters = motion_generator.step(batch_size=3)
# the `step()` function returns a dictionary:
print(filters.keys())
plot(
    [f for f in filters["filter"]],
    suptitle="Examples of randomly generated motion blurs",
)

# %%
# Other options, such as the regularity and length of the blur trajectory can also be specified:
motion_generator = MotionBlurGenerator(
    (psf_size, psf_size), l=0.6, sigma=1, device=device, dtype=dtype
)
filters = motion_generator.step(batch_size=3)
plot([f for f in filters["filter"]], suptitle="Different length and regularity")

# %%
# Diffraction blur generators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We also implemented diffraction blurs obtained through Fresnel theory and definition of the psf through the pupil
# plane expanded in Zernike polynomials


diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), device=device, dtype=dtype
)

# %%
# Then, to generate new filters, it suffices to call the step() function as follows:

filters = diffraction_generator.step(batch_size=3)

# %%
# In this case, the `step()` function returns a dictionary containing the filters,
# their pupil function and Zernike coefficients:
print(filters.keys())

# Note that we use **0.2 to increase the image dynamics
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="Examples of randomly generated diffraction blurs",
)
plot(
    [
        f
        for f in torch.angle(filters["pupil"][:, None])
        * torch.abs(filters["pupil"][:, None])
    ],
    suptitle="Corresponding pupil phases",
)
print("Coefficients of the decomposition on Zernike polynomials")
print(filters["coeff"])

# %%
# We can change the cutoff frequency (below 1/4 to respect Shannon's sampling theorem)
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=1 / 8, device=device, dtype=dtype
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="A different cutoff frequency",
)

# %%
# It is also possible to directly specify the Zernike decomposition.
# For instance, if the pupil is null, the PSF is the Airy pattern
n_zernike = len(
    diffraction_generator.list_param
)  # number of Zernike coefficients in the decomposition
filters = diffraction_generator.step(coeff=torch.zeros(3, n_zernike))
plot(
    [f for f in filters["filter"][:, None] ** 0.3],
    suptitle="Airy pattern",
)

# %%
# Finally, notice that you can activate the aberrations you want in the ANSI
# nomenclature https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=1 / 8, list_param=["Z5", "Z6"], device=device, dtype=dtype
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="PSF obtained with astigmatism only",
)

# %%
# Generator Mixture
# ~~~~~~~~~~~~~~~~~
#
# During training, it's more robust to train on multiple family of operators. This can be done
# seamlessly with the :class:`deepinv.physics.generator.GeneratorMixture`.


torch.cuda.manual_seed(4)
torch.manual_seed(6)

generator = GeneratorMixture(
    ([motion_generator, diffraction_generator]), probs=[0.5, 0.5]
)
for i in range(4):
    filters = generator.step(batch_size=3)
    plot(
        [f for f in filters["filter"]],
        suptitle=f"Random PSF generated at step {i + 1}",
    )

# %%
# Space varying blurs
# --------------------
#
# Space varying blurs are also available using :class:`deepinv.physics.SpaceVaryingBlur`


psf_size = 32
img_size = (256, 256)
n_eigenpsf = 10
spacing = (64, 64)
padding = "valid"
batch_size = 1
delta = 16

# We first instantiate a psf generator
psf_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), device=device, dtype=dtype
)
# Now, scattered random psfs are synthesized and interpolated spatially
pc_generator = ProductConvolutionBlurGenerator(
    psf_generator=psf_generator,
    image_size=img_size,
    n_eigen_psf=n_eigenpsf,
    spacing=spacing,
    padding=padding,
)
params_pc = pc_generator.step(batch_size)

physics = SpaceVaryingBlur(method="product_convolution2d", **params_pc)

dirac_comb = torch.zeros(img_size)[None, None]
dirac_comb[0, 0, ::delta, ::delta] = 1
psf_grid = physics(dirac_comb)
plot(psf_grid, titles="Space varying impulse responses")

# num_patches = 4
# i = torch.randint(
#     0,
#     img_size[0],
#     (
#         dirac_comb.size(0),
#         num_patches,
#     ),
# )
# j = torch.randint(
#     0,
#     img_size[1],
#     (
#         dirac_comb.size(0),
#         num_patches,
#     ),
# )
# centers = torch.stack((i, j), dim=-1)
# psf = physics.get_psf(centers=centers)
# plot(psf.flatten(0, 1))




# # %%
# img_size = 512
# patch_size = 128
# overlap = 64
# psf_size = 31

# psf_generator = DiffractionBlurGenerator(
#     (psf_size, psf_size), device=device, dtype=dtype
# )
# patch_psf_generator = ProductConvolutionPatchBlurGenerator(
#     psf_generator=psf_generator,
#     image_size=img_size,
#     patch_size=patch_size,
#     overlap=overlap,
# )
# patch_info = {
#     "patch_size": patch_size,
#     "overlap": overlap,
#     "num_patches": compute_patch_info(img_size, patch_size, overlap)["num_patches"],
# }
# params_pc = patch_psf_generator.step(batch_size)

# patch_physics = SpaceVaryingBlur(
#     method="product_convolution2d_patch", patch_info=patch_info, **params_pc
# )

# dirac_comb = torch.zeros(1, 1, img_size, img_size)
# delta = 2 * psf_size
# dirac_comb[0, 0, ::delta, ::delta] = 1

# psf_grid = patch_physics(dirac_comb)
# plot(psf_grid**0.1, titles="Space varying impulse responses")

# # %%
# num_patches = 4
# i = torch.randint(
#     patch_size,
#     img_size - patch_size,
#     (
#         dirac_comb.size(0),
#         num_patches,
#     ),
# )
# j = torch.randint(
#     patch_size,
#     img_size - patch_size,
#     (
#         dirac_comb.size(0),
#         num_patches,
#     ),
# )
# centers = torch.stack((i, j), dim=-1)
# psf = patch_physics.get_psf(centers=centers)
# plot(psf.flatten(0, 1) ** 0.5)
# # %%


# def generate_random_patch(tensor, centers, patch_size: tuple[int]):
#     """
#     Args:
#         tensor (Tensor): Tensor of patch_size (B, C, H, W) to be cropped.
#         centers (Tensor): (B, num_patches, 2)
#     Returns:
#         Tensor: Randomly cropped Tensor of shape (num_patches, B, C, patch_size, patch_size)
#     """
#     if isinstance(patch_size, int):
#         patch_size = (patch_size, patch_size)

#     if centers.size(0) == 1:
#         centers = centers.expand(tensor.size(0), -1, -1)
#     centers[..., 0].clamp_(patch_size[0] // 2, tensor.size(-2) - patch_size[0] // 2)
#     centers[..., 1].clamp_(patch_size[1] // 2, tensor.size(-1) - patch_size[1] // 2)
#     random_patch = []

#     for b in range(tensor.size(0)):
#         for k in range(centers.size(1)):
#             position = centers[b, k, :]
#             ih, iw = patch_size[0] % 2, patch_size[1] % 2
#             random_patch.append(
#                 tensor[
#                     b : b + 1,
#                     :,
#                     position[0]
#                     - patch_size[0] // 2 : position[0]
#                     + patch_size[0] // 2
#                     + ih,
#                     position[1]
#                     - patch_size[1] // 2 : position[1]
#                     + patch_size[1] // 2
#                     + iw,
#                 ]
#             )
#     return torch.stack(random_patch, dim=0)


# centers = torch.stack(
#     (torch.arange(0, img_size, delta), torch.arange(0, img_size, delta)), dim=1
# )[None, 1:-1, :]
# psf_grid_padded = torch.nn.functional.pad(
#     psf_grid,
#     (psf_size // 2, psf_size // 2, psf_size // 2, psf_size // 2),
#     mode="constant",
#     value=0,
# )

# psf = patch_physics.get_psf(centers)
# plot(psf.flatten(0, 1) ** 0.5)

# psf_hat = generate_random_patch(psf_grid_padded, centers, patch_size=psf_size)
# plot(psf_hat.flatten(0, 1) ** 0.5)
# plot((psf_hat - psf).flatten(0, 1) ** 0.5)

# # %%




#%%
list_params = [
    "Z4"]

diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=1 / 8, list_param=list_params, device=device, dtype=dtype
)

batch_size = 1

coeff = torch.zeros(batch_size, n_zernike)
coeff[:,0] = 0.3
filters = diffraction_generator.step(coeff=coeff)

plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="defocus only",
)


#%%
psf_size = 51

list_params = [
    "Z4",
    "Z5",
    "Z6",
    "Z7",
    "Z8",
    "Z9",
    "Z10",
    "Z11",
]
    # "Z12",
    # "Z13",
    # "Z14",
    # "Z15",    
    # "Z16",
    # "Z17",
    # "Z18",
    # "Z19",

diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=0.125, list_param=list_params, device=device, dtype=dtype
)


#%% 
import matplotlib.pyplot as plt

import torch
import torch.fft

def rotate_image_via_shear(image, angle_deg: torch.Tensor, center=None):
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
        freq_0 = shear[:, None] * (torch.arange(N0) - center[0])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[..., None] * freq_1[None, None, :])
        image_shear = fft1 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-1)))
    
    def sheary(image, shear):
        fft0 = torch.fft.fft2(image, dim=(-2))
        freq_0 = torch.fft.fftfreq(N0, d=1.0, device=image.device)
        freq_1 = shear[:, None] * (torch.arange(N1) - center[1])[None]
        phase_shift = torch.exp(-2j * torch.pi * freq_0[None, :, None] * freq_1[:, None, :])
        image_shear = fft0 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft2(image_shear, dim=(-2)))
        
    rot = shearx(sheary(shearx(transformed_image, tant2), st), tant2)
    return rot 
# # Example usage
# image = torch.zeros((128,128))  # A random 128x128 image
# image[50:78, 50:78] = 1
# angle = torch.tensor([10])  # Rotate by 45 degrees

# rotated_image = image
# for i in range(100):
#     plt.imshow(rotated_image);plt.show()
#     rotated_image = rotate_image_via_shear(rotated_image, angle)
 
#%% Random Uniform to see large Zernikes !
#from display_utils import show_images

fc = 0.2

diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=fc, list_param=list_params, apodize=True, device=device, dtype=dtype
)



batch_size = 8 * 10

n_zernike = len(list_params)

##Pierre multinomial like
#sparsity = torch.ones(n_zernike) * 1
# sparsity = torch.tensor([5., 1., 1., 1., 1., 1., 1., 1.])
# rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 2
# rand_sgn = torch.sign(rand_ampl)
# coeff = rand_sgn * torch.abs(rand_ampl)**sparsity
# coeff[:, 1:] *=  0.15
# coeff[:, 0] *=  0.5

#True multinomial
zernike_orders = get_zernike_order(diffraction_generator.index_params)
sparsity = zernike_orders - 1
sparsity[diffraction_generator.index_params == 4] = 5 #rare defocus
sparsity[diffraction_generator.index_params == 12] = 1 #often primary spherical

#sparsity = torch.tensor([5., 1., 1., 2., 2., 2., 2., 1.])
sparsity = 1.0 / sparsity
sparsity /= sparsity.sum()
multinom = torch.multinomial(sparsity.expand(batch_size, -1), num_samples=len(sparsity), replacement=True)

multinom_mask = torch.zeros_like(multinom)
for b in range(batch_size):
    multinom_mask[b] = torch.isin(torch.arange(len(sparsity)), multinom[b]).to(dtype)

rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 2
#rand_sgn = torch.sign(rand_ampl)
coeff = rand_ampl * multinom_mask
coeff[:, 1:] *=  0.15
coeff[:, 0] *=  0.4

##Random uniform 0.15 amplitude
# rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 1
# coeff = rand_ampl * 0.15


# ## Multinomail + Large normal
# sparsity = torch.tensor([5., 1., 1., 1., 1., 1., 1., 1.])
# sparsity = 1.0 / sparsity
# sparsity /= sparsity.sum()
# multinom = torch.multinomial(sparsity.expand(batch_size, -1), num_samples=len(sparsity), replacement=True)

# multinom_mask = torch.zeros_like(multinom)
# for b in range(batch_size):
#     multinom_mask[b] = torch.isin(torch.arange(len(sparsity)), multinom[b]).to(dtype)

# rand_ampl = (torch.randn(batch_size, n_zernike))
# #rand_sgn = torch.sign(rand_ampl)
# coeff = rand_ampl * multinom_mask
# coeff[:, 1:] *=  0.1
# coeff[:, 0] *=  0.3

angles_deg = torch.rand(batch_size) * 360

filters = diffraction_generator.step(coeff=coeff)["filter"]
filters_r = rotate_image_via_shear(filters, angles_deg)

#show_images(filters**0.5, ncols=8)
show_images(filters_r**0.5, ncols=8)

torch.sum(torch.sum(torch.abs(coeff) < 0.025, axis=1) > 7) / batch_size

#%%
import torch

def get_zernike_order(n):
    # Convert n to a tensor if it's not already
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, dtype=torch.float32)
    
    # Adjust n by subtracting 1 to account for 1-based indexing
    n_adjusted = n - 1
    
    # Calculate k using the quadratic formula
    k = torch.floor((-1 + torch.sqrt(1 + 8 * n_adjusted)) / 2)
    
    return k.long()

# Example usage:
n = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12])
k = find_k(n)
print(k)

#%% Add some low Zernike power PSFs for well calibrated images

diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=fc, list_param=list_params, apodize=True, device=device, dtype=dtype
)

batch_size = 100

n_zernike = len(list_params)

##Pierre multinomial like
#sparsity = torch.ones(n_zernike) * 1
# sparsity = torch.tensor([5., 1., 1., 1., 1., 1., 1., 1.])
# rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 2
# rand_sgn = torch.sign(rand_ampl)
# coeff = rand_sgn * torch.abs(rand_ampl)**sparsity
# coeff[:, 1:] *=  0.15
# coeff[:, 0] *=  0.5

# True multinomial
# sparsity = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1.])
# sparsity = 1.0 / sparsity
# sparsity /= sparsity.sum()
# multinom = torch.multinomial(sparsity.expand(batch_size, -1), num_samples=len(sparsity), replacement=True)

# multinom_mask = torch.zeros_like(multinom)
# for b in range(batch_size):
#     multinom_mask[b] = torch.isin(torch.arange(len(sparsity)), multinom[b]).to(dtype)

rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 2
#rand_sgn = torch.sign(rand_ampl)
coeff_airy = rand_ampl #* multinom_mask
coeff_airy[:, 1:] *=  0.03
coeff_airy[:, 0] *=  0.03

##Random uniform 0.15 amplitude
# rand_ampl = (torch.rand(batch_size, n_zernike) - 0.5) * 1
# coeff = rand_ampl * 0.15


# ## Multinomail + Large normal
# sparsity = torch.tensor([5., 1., 1., 1., 1., 1., 1., 1.])
# sparsity = 1.0 / sparsity
# sparsity /= sparsity.sum()
# multinom = torch.multinomial(sparsity.expand(batch_size, -1), num_samples=len(sparsity), replacement=True)

# multinom_mask = torch.zeros_like(multinom)
# for b in range(batch_size):
#     multinom_mask[b] = torch.isin(torch.arange(len(sparsity)), multinom[b]).to(dtype)

# rand_ampl = (torch.randn(batch_size, n_zernike))
# #rand_sgn = torch.sign(rand_ampl)
# coeff = rand_ampl * multinom_mask
# coeff[:, 1:] *=  0.1
# coeff[:, 0] *=  0.3

angles_deg = torch.rand(batch_size) * 360

filters = diffraction_generator.step(coeff=coeff_airy)["filter"]
filters_r_airy = rotate_image_via_shear(filters, angles_deg)

#show_images(filters**0.5, ncols=8)
#show_images(filters_r**0.5, ncols=8)

torch.sum(torch.sum(torch.abs(coeff) < 0.025, axis=1) > 7) / batch_size


filters_r_all = torch.cat([filters_r, filters_r_airy])
AU = 1.22 / (2 * fc)
power_all = filters_r_all[:, 
                      :, 
                      psf_size//2 - int(AU):psf_size//2 + int(AU), 
                      psf_size//2 - int(AU):psf_size//2 + int(AU)].sum((-2, -1)).flatten()


power_unif = filters_r[:, 
                      :, 
                      psf_size//2 - int(AU):psf_size//2 + int(AU), 
                      psf_size//2 - int(AU):psf_size//2 + int(AU)].sum((-2, -1)).flatten()

_ = plt.hist(power_all, bins=torch.linspace(0.,0.9,30));  plt.hist(power_unif, bins=torch.linspace(0.,0.9,30)); 
print('Uniform : %2.2f percent of filters have more than 80 percent power in 1 Airy Unit: ' % (100 * torch.sum(power_unif > 0.8).item() / filters_r.shape[0]))
print('Corrected : %2.2f percent of filters have more than 80 percent power in 1 Airy Unit: ' % (100 * torch.sum(power_all > 0.8).item() / filters_r_all.shape[0]))
#%%

batch_size = 2
num_patches = (7, 7)


cum_probs = torch.tensor([0.9, 1.0]).expand(batch_size * num_patches[0] * num_patches[1], -1)
p = torch.rand(batch_size * num_patches[0] * num_patches[1], 1)  # np.random.uniform()
idx = torch.searchsorted(cum_probs, p)

#True multinomial for large Zernikes
zernike_orders = get_zernike_order(diffraction_generator.index_params)
sparsity = zernike_orders - 1
sparsity[diffraction_generator.index_params == 4] = 5 #rare defocus
sparsity[diffraction_generator.index_params == 11] = 1 #often primary spherical

sparsity = 1.0 / sparsity
sparsity /= sparsity.sum()
multinom = torch.multinomial(sparsity.expand(batch_size * num_patches[0] * num_patches[1], -1), num_samples=len(sparsity), replacement=True)

multinom_mask = torch.zeros_like(multinom, device=device)
for b in range(x.size(0 * num_patches[0] * num_patches[1])):
    multinom_mask[b] = torch.isin(torch.arange(len(sparsity)), multinom[b]).to(dtype)

coeff = (torch.rand(x.size(0) * num_patches[0] * num_patches[1], generator.psf_generator.n_zernike, device=device) - 0.5) * 2

   
coeff[(idx == 0).expand(-1, generator.psf_generator.n_zernike)] *= multinom_mask[(idx == 0).expand(-1, generator.psf_generator.n_zernike)]
coeff[:, 1:][(idx == 0).expand(-1, generator.psf_generator.n_zernike - 1)] *= 0.15         
coeff[:, 0][(idx == 0)[:, 0]] *= 0.4
coeff[(idx == 1).expand(-1, generator.psf_generator.n_zernike)] *= 0.03



#%%
image = torch.zeros((4,1,128,128))  # A random 128x128 image
image[:, :, 50:78, 50:78] = 1

angles_deg = torch.tensor([45, 75, 90, 181])#torch.rand(2) * 360


cum_probs = torch.tensor([0.9, 1.0])
p = torch.rand(1).item()  # np.random.uniform()
idx = torch.searchsorted(cum_probs, p)

#%%
rotated_image = rotate_image_via_shear(image, angles_deg)


plt.imshow(image[0, 0]); plt.show()

plt.imshow(rotated_image[0, 0]); plt.show()
plt.imshow(rotated_image[1, 0]); plt.show()
plt.imshow(rotated_image[2, 0]); plt.show()
plt.imshow(rotated_image[3, 0]); plt.show()
