# %%
import math
import torch.nn.functional as F
import torch
from deepinv.physics.generator.blur import bump_function
from typing import Tuple
from torch import Tensor
from deepinv.physics.functional.multiplier import (
    multiplier,
    multiplier_adjoint,
)
from deepinv.physics.functional.convolution import conv2d, conv_transpose2d


def product_convolution2d(
    x: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> Tensor:
    r"""

    Product-convolution operator in 2d. Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions. <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
    :param padding: ( options = `valid`, `circular`, `replicate`, `reflect`. If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    :return: torch.Tensor y
    :rtype: tuple
    """

    K = w.shape[0]
    for k in range(K):
        if k == 0:
            result = conv2d(multiplier(x, w[k]), h[k], padding=padding)
        else:
            result += conv2d(multiplier(x, w[k]), h[k], padding=padding)

    return result


def get_psf_product_convolution2d(h, w, position: Tuple[int]):
    r"""
    :param torch.Tensor w: Tensor of size (K, b, c, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
    """
    return torch.sum(h * w[..., position[0]:position[0]+1, position[1]:position[1] + 1], dim=0).flip(-1).flip(-2)


# def get_psf_product_convolution2d_patches(h, w, position: Tuple[int], overlap: Tuple[int], image_size: Tuple[int]):
#     r"""
#     :param torch.Tensor w: Tensor of size (K, b, c, patch_size, patch_size). b in {1, B} and c in {1, C}
#     :param torch.Tensor h: Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
#     """
#     patch_size = w.shape[-2:]
#     patch_index, patch_position = get_all_patches_and_positions(
#         position, patch_size, overlap, image_size)

#     num_patches = ((image_size[0] - patch_size[0]) // (patch_size[0] - overlap[0]) + 1,
#                    (image_size[1] - patch_size[1]) // (patch_size[1] - overlap[1]) + 1)

#     h = h.view(num_patches[0], num_patches[1], h.size(
#         1), h.size(2), h.size(3), h.size(4))
#     w = w.view(num_patches[0], num_patches[1], w.size(
#         1), w.size(2), w.size(3), w.size(4))

#     output = 0.
#     print('xxx: ', patch_index, patch_position)
#     for index, pos in zip(patch_index, patch_position):
#         i, j = index
#         ii, jj = pos
#         print('1', i, j)
#         print('2', ii, jj)
#         output += output + h[i, j, ...] * w[i, j, ..., ii:ii+1, jj:jj+1]
#     return output.flip(-1).flip(-2) if isinstance(output, torch.Tensor) else output

def get_psf_product_convolution2d_patches(h, w, position: Tuple[int], overlap: Tuple[int], num_patches: Tuple[int]):
    r"""
    :param torch.Tensor w: Tensor of size (K, b, c, patch_size, patch_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (K, b, c, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
    """
    patch_size = w.shape[-2:]
    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    p = patch_size
    o = overlap
    n = (math.floor(position[0] / (p[0] - o[0])),
         math.floor(position[1] / (p[1] - o[1])))

    index_h = []
    index_w = []
    patch_position_h = []
    patch_position_w = []

    if n[0] == 0:
        index_h.append(n[0])
        patch_position_h.append(position[0])
    elif n[0] == num_patches[0] and position[0] < n[0] * (p[0] - o[0]) + o[0]:
        index_h.append(n[0] - 1)
        patch_position_h.append(position[0] - (n[0] - 1) * (p[0] - o[0]))

    elif n[0] > num_patches[0]:
        pass
    else:
        if position[0] <= n[0] * (p[0] - o[0]) + o[0]:
            index_h.extend([n[0] - 1, n[0]])
            patch_position_h.extend([
                position[0] - (n[0] - 1) * (p[0] - o[0]), position[0] - n[0] * (p[0] - o[0])])
        else:
            index_h.append(n[0])
            patch_position_h.append(
                position[0] - n[0] * (p[0] - o[0]))

    if n[1] == 0:
        index_w.append(n[1])
        patch_position_w.append(position[1] - n[1] * (p[1] - o[1]))
    elif n[1] == num_patches[1] and position[1] < n[1] * (p[1] - o[1]) + o[1]:
        index_w.append(n[1] - 1)
        patch_position_w.append(position[1] - (n[1] - 1) * (p[1] - o[1]))
    elif n[1] > num_patches[1] - 1:
        pass
    else:
        if position[1] <= n[1] * (p[1] - o[1]) + o[1]:
            index_w.extend([n[1] - 1, n[1]])
            patch_position_w.extend([
                position[1] - (n[1] - 1) * (p[1] - o[1]), position[1] - n[1] * (p[1] - o[1])])
        else:
            index_w.append(n[1])
            patch_position_w.append(
                position[1] - n[1] * (p[1] - o[1]))

    h = h.view(num_patches[0], num_patches[1], h.size(
        1), h.size(2), h.size(3), h.size(4))
    w = w.view(num_patches[0], num_patches[1], w.size(
        1), w.size(2), w.size(3), w.size(4))
    psf = 0.

    print('n:', n)
    print('patch_position_h', patch_position_h)
    print('patch_position_w', patch_position_w)
    print('index: ', index_h, index_w)
    for count_i, i in enumerate(index_h):
        for count_j, j in enumerate(index_w):
            psf = psf + h[i, j, ...] * w[i, j, ...,
                                         patch_position_h[count_i]:patch_position_h[count_i]+1, patch_position_w[count_j]:patch_position_w[count_j]+1]
    return psf.flip(-1).flip(-2) if isinstance(psf, torch.Tensor) else psf


def product_convolution2d_adjoint(
    y: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> Tensor:
    r"""

    Product-convolution adjoint operator in 2d.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (K, b, c, ...)
    :param torch.Tensor h: Tensor of size (K, b, c, ...)
    :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        If `padding = 'valid'` the blurred output is smaller than the image (no padding),
        otherwise the blurred output has the same size as the image.
    """

    K = w.shape[0]
    for k in range(K):
        if k == 0:
            result = multiplier_adjoint(
                conv_transpose2d(y, h[k], padding=padding), w[k]
            )
        else:
            result += multiplier_adjoint(
                conv_transpose2d(y, h[k], padding=padding), w[k]
            )

    return result


def unity_partition_function_1d(image_size, patch_size, overlap, ):
    n_patch = (image_size - patch_size) // (patch_size - overlap) + 1
    max = patch_size + (n_patch - 1) * (patch_size - overlap)
    t = torch.linspace(-max//2, max//2, max)
    mask = bump_function(t, patch_size / 2 - overlap,
                         overlap).roll(shifts=-max // 2 + patch_size//2)
    masks = torch.stack([mask.roll(shifts=(patch_size-overlap) * i)
                         for i in range(n_patch)], dim=0)
    masks[0, :overlap] = 1.
    masks[-1, -overlap:] = 1.
    masks = masks / masks.sum(dim=0, keepdims=True)
    return masks


def linear_function_1d(image_size, patch_size, overlap):
    n_patch = (image_size - patch_size) // (patch_size - overlap) + 1
    max = patch_size + (n_patch - 1) * (patch_size - overlap)
    t = torch.linspace(-max//2, max//2, max)
    a = patch_size / 2 - overlap
    b = overlap
    def f(x): return ((a + b) / b - x.abs().clip(0, a+b) / b).abs().clip(0, 1)

    mask = f(t).roll(shifts=-max // 2 + patch_size//2)
    masks = torch.stack([mask.roll(shifts=(patch_size-overlap) * i)
                         for i in range(n_patch)], dim=0)
    masks[0, :overlap] = 1.
    masks[-1, -overlap:] = 1.
    masks = masks / masks.sum(dim=0, keepdims=True)
    return masks


def unity_partition_function_2d(image_size: Tuple[int], patch_size: Tuple[int], overlap: Tuple[int]):
    r"""
    Unity partition function in 2D.
    Image will be cropped.

    """
    masks_x = unity_partition_function_1d(
        image_size[0], patch_size[0], overlap[0])
    masks_y = unity_partition_function_1d(
        image_size[1], patch_size[1], overlap[1])
    # masks_x = linear_function_1d(
    #     image_size[0], patch_size[0], overlap[0])
    # masks_y = linear_function_1d(
    #     image_size[1], patch_size[1], overlap[1])

    masks = torch.tensordot(masks_x, masks_y, dims=0)
    masks = masks.permute(0, 2, 1, 3)
    masks = masks / (masks.sum(dim=(0, 1), keepdims=True) + 1e-8)
    return masks


def crop_unity_partition_2d(masks, patch_size: Tuple[int], overlap: Tuple[int], psf_size: Tuple[int]):
    cropped_masks = []
    diff_h = patch_size[0] - overlap[0]
    diff_w = patch_size[1] - overlap[1]
    supp_h = patch_size[0]  # + psf_size[0]
    supp_w = patch_size[1]  # + psf_size[1]

    index = torch.zeros(masks.size(0), masks.size(1), 2)
    for h in range(masks.size(0)):
        for w in range(masks.size(1)):
            cropped_masks.append(
                masks[h, w, h * diff_h: h * diff_h + supp_h, w * diff_w: w * diff_w + supp_w])
            index[h, w, 0] = h * diff_h + psf_size[0] // 2
            index[h, w, 1] = w * diff_w + psf_size[1] // 2

    cropped_masks = torch.stack(cropped_masks, dim=0)
    return cropped_masks, index


def image_to_patches(image, patch_size, overlap):
    """
    Splits an image into patches with specified patch size and overlap.

    Parameters:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        patch_size (int): Size of each patch (patch_size x patch_size).
        overlap (int): Overlapping size between patches.

    Returns:
        torch.Tensor: Batch of patches of shape (B, C, n_rows, n_cols, patch_size, patch_size).
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    # Ensure image is a tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image should be a torch.Tensor")

    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])

    # Ensure the patch size and overlap are valid
    assert (stride[0] > 0) * (stride[1] >
                              0), "Patch size must be greater than overlap"

    patches = image.unfold(2, patch_size[0], stride[0]
                           ).unfold(3, patch_size[1], stride[1])
    # patches = patches.flatten(1, 2).permute(2, 0, 1, 3, 4).contiguous()
    return patches.contiguous()


def patches_to_image(patches, overlap):
    """
    Reconstruct a batch of images from patches

    Parameters:
        patches (torch.Tensor): Input image tensor of shape (B, C, n_rows, n_cols, patch_size, patch_size)..
        patch_size (int): Size of each patch (patch_size x patch_size).
        overlap (int): Overlapping size between patches.

    Returns:
        torch.Tensor: Batch of image of shape (B, C, H, W).
    """

    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    B, C, num_patches_h, num_patches_w, h, w = patches.size()

    output_size = (h + (num_patches_h - 1) * (h - overlap[0]),
                   w + (num_patches_w - 1) * (w - overlap[1]))
    print(output_size)
    if not isinstance(patches, torch.Tensor):
        raise TypeError("Patches should be a torch.Tensor")

    # Rearrange the patches to have the desired shape (B, num_patches_h * num_patches_w, C, h, w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(
        B, num_patches_h * num_patches_w, C, h, w)

    # Now reverse the process using torch.fold

    # Calculate the number of patches
    num_patches = num_patches_h * num_patches_w

    # Reshape the patches to (B, C*h*w, num_patches)
    patches = patches.view(B, num_patches, C * h * w).permute(0, 2, 1)
    stride = (h - overlap[0], w - overlap[1])
    # Fold the patches back into the image
    output = torch.nn.functional.fold(
        patches,
        output_size=output_size,
        kernel_size=(h, w),
        stride=stride,
    )
    return output.contiguous()


# %%
def product_convolution2d_patches(
    x: Tensor,
    w: Tensor,
    h: Tensor,
    patch_size: Tuple[int] = (256, 256),
    overlap: Tuple[int] = (15, 15),
) -> Tensor:
    r"""

    Product-convolution operator in 2d. Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions. <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    The convolution is done by patches, using only 'valid' padding.
    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (K, b, c, patch_size + psf_size, patch_size + psf_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (K, b, c, psf_size, psf_size). b in {1, B} and c in {1, C}, h<=H and w<=W
    :return: torch.Tensor y
    :rtype: tuple
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    psf_size = h.shape[-2:]
    patches = image_to_patches(
        x,
        patch_size=patch_size,
        overlap=overlap)  # (B, C, K1, K2, P1, P2)

    n_rows, n_cols = patches.size(2), patches.size(3)
    assert n_rows * \
        n_cols == w.size(
            0), 'The number of patches must be equal to the number of PSFs'

    patches = patches.flatten(2, 3)
    patches = patches * w.permute(1, 2, 0, 3, 4)

    patches = F.pad(patches, pad=(
        psf_size[1] // 2, psf_size[1] // 2, psf_size[0] // 2, psf_size[0] // 2), value=0, mode='constant')

    result = []
    for k in range(h.size(0)):
        result.append(conv2d(
            patches[:, :, k, ...], h[k], padding='constant',
        ))
    # (B, C, K, H', W')

    result = torch.stack(result, dim=2)
    margin = (psf_size[0] // 2, psf_size[1] // 2)
    B, C, K, H, W = result.size()
    result = patches_to_image(result.view(
        B, C, n_rows, n_cols, H, W), add_tuple(overlap, psf_size))[..., margin[0]:-margin[0], margin[1]:-margin[1]]
    return result


def add_tuple(a, b):
    assert len(a) == len(b)
    return tuple(a[i] + b[i] for i in range(len(a)))
