"""HRNet-W40 backbone, trimmed to the feature-extraction trunk the released
checkpoint actually uses.

The legacy backbone (``lib/models/backbones/hrnet.py::HRNet``) builds a
HighResolutionNet with its full ImageNet classification head (``incre_modules``,
``downsamp_modules``, ``final_layer``, ``classifier``) even though
``forward()`` never calls it -- it returns the four stage-4 branch feature
maps directly. Those classifier weights still exist in the released
checkpoint (trained once upstream of this model) but are dead weight for
every number in the paper. This module keeps only the trunk that is actually
executed: ``conv1/bn1/conv2/bn2 -> layer1 -> (transition+stage) x3``.

Attribute names mirror the legacy module 1:1 (``conv1``, ``bn1``, ``layer1``,
``transitionN``, ``stageN``, ...) so ``load_state_dict(strict=False)`` on the
released ``img_backbone.*`` checkpoint keys populates every weight here; the
classifier-head keys are simply absent from this module and are reported as
ignored by the parity test.
"""
from typing import List

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1

# HRNet-W40 stage config (config/backbone/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml).
# channels are post-expansion (BASIC block expansion = 1).
_STAGE1 = dict(num_modules=1, num_branches=1, num_blocks=[4], num_channels=[64], block="BOTTLENECK")
_STAGE2 = dict(num_modules=1, num_branches=2, num_blocks=[4, 4], num_channels=[40, 80], block="BASIC")
_STAGE3 = dict(num_modules=4, num_branches=3, num_blocks=[4, 4, 4], num_channels=[40, 80, 160], block="BASIC")
_STAGE4 = dict(num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[40, 80, 160, 320],
                block="BASIC")

# Output channels per branch, high-res -> low-res (what the neck consumes).
W40_OUT_CHANNELS = [40, 80, 160, 320]


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


_BLOCKS = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    """One HRNet fusion module: parallel per-branch residual blocks + a dense
    top-down/bottom-up fuse across branches. Identical to the legacy module
    (kept verbatim; this part of HRNet is already minimal)."""

    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels,
                 multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _make_one_branch(self, idx, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[idx] != num_channels[idx] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[idx], num_channels[idx] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[idx] * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(self.num_inchannels[idx], num_channels[idx], stride, downsample)]
        self.num_inchannels[idx] = num_channels[idx] * block.expansion
        for _ in range(1, num_blocks[idx]):
            layers.append(block(self.num_inchannels[idx], num_channels[idx]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels)
                              for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest")))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = num_inchannels[i] if last else num_inchannels[j]
                        block = [nn.Conv2d(num_inchannels[j], out_ch, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)]
                        if not last:
                            block.append(nn.ReLU(False))
                        conv3x3s.append(nn.Sequential(*block))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        x = [self.branches[i](xi) for i, xi in enumerate(x)]
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


def _make_transition_layer(num_channels_pre, num_channels_cur):
    num_pre, num_cur = len(num_channels_pre), len(num_channels_cur)
    layers = []
    for i in range(num_cur):
        if i < num_pre:
            if num_channels_cur[i] != num_channels_pre[i]:
                layers.append(nn.Sequential(
                    nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
            else:
                layers.append(None)
        else:
            conv3x3s = []
            for j in range(i + 1 - num_pre):
                in_ch = num_channels_pre[-1]
                out_ch = num_channels_cur[i] if j == i - num_pre else in_ch
                conv3x3s.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
            layers.append(nn.Sequential(*conv3x3s))
    return nn.ModuleList(layers)


def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


def _make_stage(cfg, num_inchannels, multi_scale_output=True):
    block = _BLOCKS[cfg["block"]]
    modules = []
    for i in range(cfg["num_modules"]):
        last = i == cfg["num_modules"] - 1
        modules.append(HighResolutionModule(
            cfg["num_branches"], block, cfg["num_blocks"], num_inchannels, cfg["num_channels"],
            multi_scale_output=(True if not last else multi_scale_output)))
        num_inchannels = modules[-1].num_inchannels
    return nn.Sequential(*modules), num_inchannels


class HRNetW40(nn.Module):
    """HRNet-W40 trunk (stem + 4 stages), no classification head.

    forward(images) -> list of 4 feature maps, high-res -> low-res, with
    channel counts ``W40_OUT_CHANNELS`` and strides [4, 8, 16, 32] (for a
    256x256 input: 64x64, 32x32, 16x16, 8x8).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        block1 = _BLOCKS[_STAGE1["block"]]
        self.layer1 = _make_layer(block1, 64, _STAGE1["num_channels"][0], _STAGE1["num_blocks"][0])
        stage1_out = block1.expansion * _STAGE1["num_channels"][0]

        block2 = _BLOCKS[_STAGE2["block"]]
        ch2 = [c * block2.expansion for c in _STAGE2["num_channels"]]
        self.transition1 = _make_transition_layer([stage1_out], ch2)
        self.stage2, pre_ch = _make_stage(_STAGE2, list(ch2))

        block3 = _BLOCKS[_STAGE3["block"]]
        ch3 = [c * block3.expansion for c in _STAGE3["num_channels"]]
        self.transition2 = _make_transition_layer(pre_ch, ch3)
        self.stage3, pre_ch = _make_stage(_STAGE3, list(ch3))

        block4 = _BLOCKS[_STAGE4["block"]]
        ch4 = [c * block4.expansion for c in _STAGE4["num_channels"]]
        self.transition3 = _make_transition_layer(pre_ch, ch4)
        self.stage4, _ = _make_stage(_STAGE4, list(ch4), multi_scale_output=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x
                  for i in range(_STAGE2["num_branches"])]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i]
                  for i in range(_STAGE3["num_branches"])]
        y_list = self.stage3(x_list)

        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i]
                  for i in range(_STAGE4["num_branches"])]
        y_list = self.stage4(x_list)
        return y_list
