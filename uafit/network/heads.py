"""Neck + per-view heads: feature fusion, heatmap, dense vote field, per-vote
uncertainty, and the two small heads that feed the MANO shape ("beta_hat").

All classes mirror the legacy ``lib/models/dovf/dovf_decoder.py`` /
``uncertainty_modules.py`` modules attribute-for-attribute so the released
checkpoint loads directly; the dead ``DOVF_HEAD_CONTEXT`` context modules
(identity for this checkpoint -- ``DOVF_HEAD_CONTEXT`` was never set to
anything but the default "none") and the coarse-to-fine *per-scale* outputs
(used only for a training-time deep-supervision loss) are dropped.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Neck: top-down FPN fusing the 4 HRNet branches into a common feat_dim
# ──────────────────────────────────────────────────────────────────────────

class MultiResNeck(nn.Module):
    """Top-down FPN. in/out feature lists are ordered high-res -> low-res."""

    def __init__(self, in_channels_list: List[int], feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.lateral = nn.ModuleList(nn.Conv2d(c, feat_dim, 1) for c in in_channels_list)
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
                          nn.GroupNorm(min(32, feat_dim), feat_dim), nn.ReLU(inplace=True))
            for _ in in_channels_list)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up
        return [self.smooth[i](laterals[i]) for i in range(len(laterals))]


# ──────────────────────────────────────────────────────────────────────────
# Heatmap head: per-joint softmax heatmap -> soft-argmax 2D coords
# ──────────────────────────────────────────────────────────────────────────

class HeatmapHead(nn.Module):
    """Per-joint heatmap on the finest pyramid level -> probs + soft-argmax coords."""

    def __init__(self, feat_dim: int, num_joints: int, heatmap_size: Tuple[int, int]):
        super().__init__()
        self.num_joints = num_joints
        self.hm_h, self.hm_w = heatmap_size
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, num_joints, 1))

    def forward(self, feat: torch.Tensor):
        logits = self.conv(feat)                                     # (BN, J, h, w)
        if logits.shape[-2:] != (self.hm_h, self.hm_w):
            logits = F.interpolate(logits, size=(self.hm_h, self.hm_w),
                                   mode="bilinear", align_corners=False)
        b, j, h, w = logits.shape
        probs = F.softmax(logits.flatten(2), dim=-1).view(b, j, h, w)
        xs = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)                          # (h, w, 2)
        coords = torch.einsum("bjhw,hwc->bjc", probs, grid)           # (BN, J, 2)
        return logits, probs, coords


# ──────────────────────────────────────────────────────────────────────────
# Dense vote (DOVF) field head: coarse-to-fine per-pixel offset to each joint
# ──────────────────────────────────────────────────────────────────────────

class CoarseToFineDOVFHead(nn.Module):
    """Dense per-pixel offset field, built coarse-to-fine across the pyramid.

    Only the final (heatmap-resolution) field is returned; the legacy
    per-scale outputs exist solely for a training-time deep-supervision
    loss and are not part of the eval forward path.
    """

    def __init__(self, feat_dim: int, num_joints: int, heatmap_size: Tuple[int, int], n_levels: int):
        super().__init__()
        self.num_joints = num_joints
        self.hm_h, self.hm_w = heatmap_size
        self.out_ch = num_joints * 2
        self._n_levels = n_levels
        self.coarse = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, self.out_ch, 1))
        self.res_heads = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(inplace=True),
                          nn.Conv2d(feat_dim, self.out_ch, 1))
            for _ in range(n_levels - 1))

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """pyramid: high-res -> low-res. Returns field (BN, J, 2, hm_h, hm_w)."""
        assert len(pyramid) == self._n_levels
        levels = pyramid[::-1]                                        # coarsest -> finest
        field = self.coarse(levels[0])
        for i in range(1, len(levels)):
            field_up = F.interpolate(field, size=levels[i].shape[-2:],
                                     mode="bilinear", align_corners=False)
            field = field_up + self.res_heads[i - 1](levels[i])
        if field.shape[-2:] != (self.hm_h, self.hm_w):
            field = F.interpolate(field, size=(self.hm_h, self.hm_w),
                                  mode="bilinear", align_corners=False)
        b = field.shape[0]
        return field.view(b, self.num_joints, 2, self.hm_h, self.hm_w)


# ──────────────────────────────────────────────────────────────────────────
# Per-vote uncertainty head: dense Cholesky field co-located with the DOVF field
# ──────────────────────────────────────────────────────────────────────────

class UncertaintyFieldHead(nn.Module):
    """High-res neck feature -> per-joint raw Cholesky field (BN, J, 3, h, w).

    The 3 channels are the raw (unconstrained) params of a lower-triangular
    2x2 Cholesky factor; see :func:`uafit.network.consensus.chol_to_prec`
    for how they turn into a 2x2 precision matrix.
    """

    def __init__(self, feat_dim: int, num_joints: int, hidden: int = None, groups: int = 8):
        super().__init__()
        hidden = hidden or feat_dim
        self.n_pts = num_joints
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 3, padding=1),
            nn.GroupNorm(min(groups, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_joints * 3, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        o = self.net(x)
        h, w = o.shape[-2:]
        return o.view(b, self.n_pts, 3, h, w)


# ──────────────────────────────────────────────────────────────────────────
# MANO init head: pooled per-scene descriptor -> coarse pose (unused at eval,
# kept so the checkpoint's shared trunk+pose_head weights load) + shape
# ──────────────────────────────────────────────────────────────────────────

class ManoInitHead(nn.Module):
    """Regress a coarse MANO pose (48) and shape/betas (10) from a descriptor.

    At eval, only the betas output feeds :class:`uafit.network.network.UAFitNetwork`
    (the feed-forward shape head); ``pose`` is produced because the trunk is
    shared with the checkpoint's ``init_head.pose_head``, not because the
    network's joint prediction uses it (DOVFManoMVEpi's own pose init is a
    parameter-free Kabsch alignment, done by the fitter, not this head).
    """

    def __init__(self, feat_dim: int, n_pose: int = 48, n_betas: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True))
        self.pose_head = nn.Linear(feat_dim, n_pose)
        self.betas_head = nn.Linear(feat_dim, n_betas)

    def forward(self, desc: torch.Tensor):
        x = self.mlp(desc)
        return self.pose_head(x), self.betas_head(x)


# ──────────────────────────────────────────────────────────────────────────
# Shape-residual MLP: per-scene MANO bone lengths -> additive beta residual
# ──────────────────────────────────────────────────────────────────────────

def build_beta_bone_mlp() -> nn.Sequential:
    """20 OpenPose-21 bone lengths (scaled x10) -> additive (10,) beta residual.

    Matches the checkpoint's top-level ``beta_bone_mlp`` attribute (EPI
    config ``BETA_BONE_FEAT=True``). Fixes the feed-forward shape head's
    regression-to-mean shrinkage: the pooled image descriptor alone is
    weakly shape-informative, while the triangulated joints' bone lengths
    span the beta subspace that drives MPVPE.
    """
    return nn.Sequential(
        nn.Linear(20, 128), nn.ReLU(inplace=True),
        nn.Linear(128, 128), nn.ReLU(inplace=True),
        nn.Linear(128, 10))


# OpenPose-21 bone topology (child, parent) used by build_beta_bone_mlp's input.
BONE_CHILD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
BONE_PARENT = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
