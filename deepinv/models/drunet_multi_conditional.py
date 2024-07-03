# Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models

import torch
import torch.nn as nn

import deepinv as dinv
from deepinv.models.utils import get_weights_url, test_onesplit, test_pad

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

from .blocks import CondResBlock, NoiseEmbedding


class DRUNetConditional(nn.Module):
    r"""
    DRUNet denoiser network.

    The network architecture is based on the paper
    `Learning deep CNN denoiser prior for image restoration <https://arxiv.org/abs/1704.03264>`_,
    and has a U-Net like structure, with convolutional blocks in the encoder and decoder parts.

    The network takes into account the noise level of the input image, which is encoded as an additional input channel.

    A pretrained network for (in_channels=out_channels=1 or in_channels=out_channels=3)
    can be downloaded via setting ``pretrained='download'``.

    :param int in_channels: number of channels of the input.
    :param int out_channels: number of channels of the output.
    :param list nc: number of convolutional layers.
    :param int nb: number of convolutional blocks per layer.
    :param int nf: number of channels per convolutional layer.
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU "E" for ELU and "S" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelsuffle" for pixel
        shuffling, and "upconv" for nearest neighbour upsampling with additional convolution.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param bool train: training or testing mode.
    :param str device: gpu or cpu.

    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        nc=[64, 128, 256, 512],
        nb=4,
        pretrained=None,
        train=False,
        device=None,
    ):
        super(DRUNetConditional, self).__init__()

        self.noise_embedding = NoiseEmbedding(num_channels=in_channels, emb_channels=max(nc), device=device)

        self.m_head = conv(in_channels, nc[0], bias=False, mode="C")

        self.m_down1 = ConditionalDownsampleBlock(nc[0], nc[1], bias=False, nb=nb)
        self.m_down2 = ConditionalDownsampleBlock(nc[1], nc[2], bias=False, nb=nb)
        self.m_down3 = ConditionalDownsampleBlock(nc[2], nc[3], bias=False, nb=nb)

        self.m_body = CondResBlock(nc[3], nc[3], bias=False)

        self.m_up3 = ConditionalUpsampleBlock(nc[3], nc[2], bias=False, nb=nb)
        self.m_up2 = ConditionalUpsampleBlock(nc[2], nc[1], bias=False, nb=nb)
        self.m_up1 = ConditionalUpsampleBlock(nc[1], nc[0], bias=False, nb=nb)

        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C")

        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 4:
                    name = "drunet_deepinv_color_finetune_22k.pth"
                elif in_channels == 2:
                    name = "drunet_deepinv_gray_finetune_26k.pth"
                url = get_weights_url(model_name="drunet", file_name=name)
                ckpt_drunet = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt_drunet = torch.load(
                    pretrained, map_location=lambda storage, loc: storage
                )

            self.load_state_dict(ckpt_drunet, strict=True)
        else:
            self.apply(weights_init_drunet)

        if not train:
            self.eval()
            for _, v in self.named_parameters():
                v.requires_grad = False

        if device is not None:
            self.to(device)

    def forward_unet(self, x0, sigma):
        emb_sigma = self.noise_embedding(sigma)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1, emb_sigma=emb_sigma)
        x3 = self.m_down2(x2, emb_sigma=emb_sigma)
        x4 = self.m_down3(x3, emb_sigma=emb_sigma)
        x = self.m_body(x4, emb_sigma=emb_sigma)
        x = self.m_up3(x + x4, emb_sigma=emb_sigma)
        x = self.m_up2(x + x3, emb_sigma=emb_sigma)
        x = self.m_up1(x + x2, emb_sigma=emb_sigma)
        x = self.m_tail(x + x1)
        return x

    def forward(self, x, sigma):
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float, torch.Tensor sigma: noise level. If ``sigma`` is a float, it is used for all images in the batch.
            If ``sigma`` is a tensor, it must be of shape ``(batch_size,)``.
        """
        if x.shape[1] < 3: # zero-pad the input image to 3 channels
            self.padded = True
            self.xshape1 = x.shape[1]
            pad = torch.zeros(x.shape[0], 3 - x.shape[1], x.shape[2], x.shape[3], device=x.device)
            x = torch.cat((x, pad), 1)
        else:
            self.padded = False
        x = self.forward_unet(x, torch.tensor([sigma]))
        if self.padded:
            x = x[:, :self.xshape1, :, :]
        return x


"""
Functional blocks below
"""
from collections import OrderedDict
import torch
import torch.nn as nn


"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



class ConditionalDownsampleBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, nb=4):
        super(ConditionalDownsampleBlock, self).__init__()

        self.cond_resblock_0 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.cond_resblock_1 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.cond_resblock_2 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.downsample = downsample_strideconv(in_channels, out_channels, bias=bias, mode="2")

    def forward(self, x, emb_sigma):
        x = self.cond_resblock_0(x, emb_sigma)
        x = self.cond_resblock_1(x, emb_sigma)
        x = self.cond_resblock_2(x, emb_sigma)
        x = self.downsample(x)
        return x


class ConditionalUpsampleBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, nb=4):
        super(ConditionalUpsampleBlock, self).__init__()

        self.cond_resblock_0 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.cond_resblock_1 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.cond_resblock_2 = CondResBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        self.upsample = upsample_convtranspose(in_channels, out_channels, bias=bias, mode="2")

    def forward(self, x, emb_sigma):
        x = self.cond_resblock_0(x, emb_sigma)
        x = self.cond_resblock_1(x, emb_sigma)
        x = self.cond_resblock_2(x, emb_sigma)
        x = self.upsample(x)
        return x


"""
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
"""


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
    cond_flag=False,
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "E":
            L.append(nn.ELU(inplace=False))
        elif t == "s":
            L.append(nn.Softplus())
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)

"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return up1


"""
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
"""


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


def weights_init_drunet(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)


if __name__ == '__main__':
    model = DRUNetConditional(pretrained=None)
    x = torch.rand(1, 3, 64, 64)
    sigma = torch.rand(1,)
    print(model.forward(x, sigma).shape)