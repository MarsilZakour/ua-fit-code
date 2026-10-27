"""Table-1 metrics (all in millimetres, per sample).

MPJPE compares *mesh-regressed* joints on both sides: the ground-truth joints are
regressed from the ground-truth MANO mesh with the same regressor that defines
the predicted joints (16 MANO joints + 5 fingertip vertices, OpenPose order),
because the datasets' raw joint annotations follow different conventions.
"""
from __future__ import annotations

import torch
from torch import Tensor

#: MANO fingertip vertices (thumb, index, middle, ring, pinky) and the OpenPose ordering of [16 joints, 5 tips].
MANO_TIP_VERTICES = [744, 320, 443, 555, 672]
OPENPOSE_FROM_MANO = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def mano_joints_from_vertices(vertices: Tensor, joint_regressor: Tensor) -> Tensor:
    """MANO vertices (B,778,3) -> 21 OpenPose-ordered joints (B,21,3); ``joint_regressor`` is MANO's (16,778)."""
    joints = torch.einsum("jv,bvk->bjk", joint_regressor.to(vertices), vertices)
    return torch.cat([joints, vertices[:, MANO_TIP_VERTICES]], dim=1)[:, OPENPOSE_FROM_MANO]


def mean_point_error(pred: Tensor, target: Tensor) -> Tensor:
    """(B,N,3) metres -> (B,) mean Euclidean error in mm."""
    return (pred - target).norm(dim=-1).mean(-1) * 1000.0


def procrustes_aligned_error(pred: Tensor, target: Tensor) -> Tensor:
    """Mean error (B,) in mm after the similarity transform (scale, rotation, translation) that best aligns pred to target."""
    mu_p, mu_t = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
    X, Y = pred - mu_p, target - mu_t
    U, S, Vh = torch.linalg.svd(X.transpose(1, 2) @ Y)
    V = Vh.transpose(1, 2)
    D = torch.diag_embed(torch.ones_like(S))
    D[:, 2, 2] = torch.sign(torch.det(V @ U.transpose(1, 2)))
    R = V @ D @ U.transpose(1, 2)
    scale = (S * torch.diagonal(D, dim1=-2, dim2=-1)).sum(-1) / (X ** 2).sum(dim=(1, 2)).clamp(min=1e-8)
    aligned = scale[:, None, None] * torch.einsum("bij,bvj->bvi", R, X) + mu_t
    return mean_point_error(aligned, target)
