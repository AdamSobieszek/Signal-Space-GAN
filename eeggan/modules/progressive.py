# coding=utf-8
from torch import nn
from eeggan.modules.layers.reshape import Reshape, PixelShuffle2d
from eeggan.modules.layers.normalization import PixelNorm
from eeggan.modules.layers.weight_scaling import weight_scale
from eeggan.modules.layers.stdmap import StdMap1d
from torch.nn.init import calculate_gain
from typing import List
import torch

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""
def checker():
    print('bop')

class ProgressiveDiscriminator(nn.Module):
    """
    Discriminator module for implementing progressive GANS

    Attributes
    ----------
    block : list
        List of `ProgressiveDiscriminatorBlock` which each represent one
        stage during progression
    cur_block : int
        Current stage of progression (from last to first)
    alpha : float
        Fading parameter. Defines how much of the input skips the current block

    Parameters
    ----------
    blocks : int
        Number of progression stages
    """

    def __init__(self, blocks):
        super(ProgressiveDiscriminator, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.

    def forward(self, input):
        fade = False
        alpha = self.alpha
        # Check everything, that is impor

        for i in range(self.cur_block, len(self.blocks)):
            if alpha < 1. and i == self.cur_block:
                tmp = self.blocks[i].fade_sequence(input)
                tmp = self.blocks[i + 1].in_sequence(tmp)
                fade = True

            if fade and i == self.cur_block + 1:
                input = alpha * input + (1. - alpha) * tmp

            input = self.blocks[i](input,
                                   first=(i == self.cur_block))
        return input

    def downsample_to_block(self, input, i_block):
        """
        Scales down input to the size of current input stage.
        Utilizes `ProgressiveDiscriminatorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        input : autograd.Variable
            Input data
        i_block : int
            Stage to which input should be downsampled

        Returns
        -------
        output : autograd.Variable
            Downsampled data
        """
        for i in range(i_block):
            input = self.blocks[i].fade_sequence(input)
        output = input
        return output


class ProgressiveGenerator(nn.Module):
    """
    Generator module for implementing progressive GANS

    Attributes
    ----------
    block : list
        List of `ProgressiveGeneratorBlock` which each represent one
        stage during progression
    cur_block : int
        Current stage of progression (from first to last)
    alpha : float
        Fading parameter. Defines how much of the second to last stage gets
        merged into the output.

    Parameters
    ----------
    blocks : int
        Number of progression stages
    """

    def __init__(self, blocks):
        super(ProgressiveGenerator, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        self.cur_block = 0
        self.alpha = 1.

    def forward(self, input):
        fade = False
        alpha = self.alpha
        for i in range(0, self.cur_block + 1):
            input = self.blocks[i](input, last=(i == self.cur_block))
            if alpha < 1. and i == self.cur_block - 1:
                tmp = self.blocks[i].out_sequence(input)
                fade = True

        if fade:
            tmp = self.blocks[i - 1].fade_sequence(tmp)
            input = alpha * input + (1. - alpha) * tmp
        return input


class ProgressiveDiscriminatorBlock(nn.Module):
    """
    Block for one Discriminator stage during progression

    Attributes
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage
    in_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current input
    fade_sequence : nn.Sequence
        Sequence of modules that is used for fading input into stage
    """

    def __init__(self, intermediate_sequence, in_sequence, fade_sequence):
        super(ProgressiveDiscriminatorBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence
        self.fade_sequence = fade_sequence

    def forward(self, input, first=False):
        if first:
            input = self.in_sequence(input)
        out = self.intermediate_sequence(input) ## error
        return out


class ProgressiveGeneratorBlock(nn.Module):
    """
    Block for one Generator stage during progression

    Attributes
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage
    out_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current output
    fade_sequence : nn.Sequence
        Sequence of modules that is used for fading stage into output
    """

    def __init__(self, intermediate_sequence, out_sequence, fade_sequence):
        super(ProgressiveGeneratorBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence
        self.fade_sequence = fade_sequence

    def forward(self, input, last=False):
        out = self.intermediate_sequence(input)
        if last:
            out = self.out_sequence(out)
        return out


class DiscriminatorBlocks(nn.Module):

    def __init__(
            self, n_blocks: int, n_chans: int,
            in_filters: int, out_filters: int, factor: int
    ):
        """Make Discriminator Blocks

        Args:
            n_blocks (int): number of blocks
            n_chans (int): number of channels
            in_filters (int): number of conv in_filters
            out_filters (int): number of conv out_filters
            factor (int): upscale/downscale factor in max/avg pooling
        """
        super(DiscriminatorBlocks).__init__()
        self.n_chans = n_chans
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.n_blocks = n_blocks
        self.factor = factor

    def create_conv_sequence(self):
        return nn.Sequential(
            weight_scale(nn.Conv1d(self.in_filters,
                                   self.in_filters,
                                   9,
                                   padding=4),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(nn.Conv1d(self.in_filters,
                                   self.out_filters,
                                   9,
                                   padding=4),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(nn.Conv1d(self.out_filters,
                                   self.out_filters,
                                   2,
                                   stride=2),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
        )

    def create_primary_conv_sequence(self):
        return nn.Sequential(
            weight_scale(nn.Conv1d(51,
                                   50,
                                   9,
                                   padding=4),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(nn.Conv1d(50,
                                   50,
                                   9,
                                   padding=4),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(nn.Conv1d(50,
                                   50,
                                   2,
                                   stride=2),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
        )


    def create_out_sequence(self):
        return nn.Sequential(
            weight_scale(nn.Conv2d(1, self.out_filters, (1, self.n_chans)),
                         gain=calculate_gain('leaky_relu')),
            Reshape([[0], [1], [2]]),
            nn.LeakyReLU(0.2))

    def create_fade_sequence(self):
        return nn.AvgPool2d((self.factor, 1), stride=(self.factor, 1))

    def get_blocks(self) -> torch.nn.Module:
        blocks = []
        for i in range(self.n_blocks):
            if i == self.n_blocks - 1:
                tmp_block = ProgressiveDiscriminatorBlock(
                    nn.Sequential(StdMap1d(),
                                  self.create_primary_conv_sequence(),
                                  # Reshape([[0], [1], [2]]),
                                  Reshape((-1, 50*12), override=True, print_shape = False),
                                  weight_scale(nn.Linear(50 * 12, 1),
                                               gain=calculate_gain('linear'))),
                    self.create_out_sequence(),
                    None)
                blocks.append(tmp_block)
                return blocks
            tmp_block = ProgressiveDiscriminatorBlock(
                self.create_conv_sequence(),
                self.create_out_sequence(),
                self.create_fade_sequence()
            )
            blocks.append(tmp_block)


class GeneratorBlocks(nn.Module):
    def __init__(
            self, n_blocks: int, n_chans: int,
            z_vars: int, in_filters: int,
            out_filters: int, factor: int
    ):
        """Make Generator Blocks

        Args:
            n_blocks (int): number of blocks
            n_chans (int): number of channels
            z_vars (int): latent dim
            in_filters (int): number of conv in_filters
            out_filters (int): number of conv out_filters
            factor (int): upscale/downscale factor in max/avg pooling
        """
        super(GeneratorBlocks).__init__()
        self.n_chans = n_chans
        self.z_vars = z_vars
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.factor = factor
        self.n_blocks = n_blocks

    def create_conv_sequence(self):
        return nn.Sequential(
            nn.Upsample(mode='linear', scale_factor=2),
            weight_scale(nn.Conv1d(self.in_filters,
                                   self.out_filters, 9, padding=4),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(nn.Conv1d(
                self.out_filters,
                self.out_filters, 9,
                padding=4
            ),
                gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

    def create_out_sequence(self):
        return nn.Sequential(
            weight_scale(nn.Conv1d(self.out_filters, self.n_chans, 1),
                         gain=calculate_gain('linear')),
            Reshape([[0], [1], [2], 1]),
            PixelShuffle2d([1, self.n_chans])
        )

    def create_fade_sequence(self):
        return nn.Upsample(mode='bilinear', scale_factor=(self.factor, 1))

    def get_blocks(self) -> torch.nn.Module:
        blocks = []
        for i in range(self.n_blocks):
            if i == 0:
                tmp_block = ProgressiveGeneratorBlock(
                    nn.Sequential(
                        weight_scale(
                            nn.Linear(self.z_vars, self.in_filters * 12),
                            gain=calculate_gain('leaky_relu')),
                        nn.LeakyReLU(0.2),
                        Reshape([[0], self.in_filters, -1]),
                        self.create_conv_sequence()),
                    self.create_out_sequence(),
                    self.create_fade_sequence()
                )
                blocks.append(tmp_block)
            tmp_block = ProgressiveGeneratorBlock(
                self.create_conv_sequence(),
                self.create_out_sequence(),
                self.create_fade_sequence()
            )
            blocks.append(tmp_block)
        return blocks
