"""The UA-Fit solver: fit a hand model to the network's multi-view evidence.

Objective (paper, Sec. 3), minimised over the hand parameters theta with
Levenberg-Marquardt (parafit ``LMSolver``):

    E(theta) = sum_{views v, joints j} rho( r_vj^T Omega_vj r_vj )        vote-field reprojection term
             + lambda_3D * sum_j |X_j(theta) - mu_j|^2                    anchor to the triangulated joints
             + pose prior                                                  (hand model specific)

* ``X_j(theta)`` are the hand's 21 MANO-convention joints (world, metres),
  projected into every view with the heatmap-scaled intrinsics.
* ``r_vj`` is the network's *vote field* of joint j sampled at that projection:
  a 2D vector pointing from the pixel to where the joint should be, so the
  residual vanishes when the projection lands on the voted location.
* ``Omega_vj`` is the 2x2 precision (inverse covariance) the network predicts
  at the same pixel (raw Cholesky parameters, ``precision_from_cholesky``).
* ``rho`` is a Huber loss, applied as iteratively re-weighted least squares,
  with the network's learned radius scaled by ``FitSettings.huber_radius_scale``.
* ``mu_j`` are the triangulated joints of the network.

The vote field is a constant of the solve: its Jacobian is the bilinear spatial
gradient of the field times the projection Jacobian times the model's landmark
Jacobian.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from parafit.core.solver import LMSolver
from parafit.core.types import GNBlock, State
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.energies.base import Energy, gauss_newton_block

from uafit.hand_models import FittableHand


@dataclass(frozen=True)
class FitSettings:
    """Solver settings; the defaults reproduce the paper (Table 1, "fast" feed-forward-shape mode)."""
    iterations: int = 15
    joint_anchor_weight: float = 3e5     # lambda_3D
    huber_radius_scale: float = 0.5      # learned radius (3 px) -> 1.5 px
    lm_damping: float = 1e-3             # initial LM damping (adapted x0.3 / x3 per accepted / rejected step)


# ---------------------------------------------------------------------------------------------- per-scene view batch
@dataclass
class SceneViews:
    """The per-view evidence of B scenes, padded to the largest view count V.

    vote_field (B,V,h,w,21,2) and precision_chol (B,V,h,w,21,3) in heatmap pixels; intrinsics (B,V,3,3)
    in heatmap pixels; world_to_camera (B,V,4,4); view_mask (B,V) = 1 for real views, 0 for padding.
    """
    vote_field: Tensor
    precision_chol: Tensor
    intrinsics: Tensor
    world_to_camera: Tensor
    view_mask: Tensor

    @property
    def heatmap_size(self) -> tuple[int, int]:
        return self.vote_field.shape[2], self.vote_field.shape[3]

    @classmethod
    def from_network(cls, evidence) -> "SceneViews":
        """Pad the flat per-view tensors of a ``NetworkEvidence`` by scene (views_per_scene)."""
        counts = [int(n) for n in evidence.views_per_scene]
        B, V = len(counts), max(counts)
        flat_field, flat_chol = evidence.vote_field.float(), evidence.vote_precision_chol.float()
        _, J, _, h, w = flat_field.shape
        dev = flat_field.device
        vote_field = flat_field.new_zeros(B, V, h, w, J, 2)
        precision_chol = flat_field.new_zeros(B, V, h, w, J, 3)
        intrinsics = flat_field.new_zeros(B, V, 3, 3)
        world_to_camera = torch.eye(4, device=dev).repeat(B, V, 1, 1)
        view_mask = flat_field.new_zeros(B, V)
        start = 0
        for i, n in enumerate(counts):
            views = slice(start, start + n)
            vote_field[i, :n] = flat_field[views].permute(0, 3, 4, 1, 2)          # (n,J,2,h,w) -> (n,h,w,J,2)
            precision_chol[i, :n] = flat_chol[views].permute(0, 3, 4, 1, 2)
            intrinsics[i, :n] = evidence.intrinsics_heatmap[views].float()
            world_to_camera[i, :n] = evidence.world_to_camera[views].float()
            view_mask[i, :n] = 1.0
            start += n
        return cls(vote_field, precision_chol, intrinsics, world_to_camera, view_mask)


# ---------------------------------------------------------------------------------------------- geometry
def project_with_jacobian(points: Tensor, intrinsics: Tensor, world_to_camera: Tensor) -> tuple[Tensor, Tensor]:
    """Pinhole projection of world points (B,J,3) into V views.

    Returns pixels (B,V,J,2) and d pixels / d world point (B,V,J,2,3). Depth is clamped at 1 mm.
    """
    R, t = world_to_camera[:, :, :3, :3], world_to_camera[:, :, :3, 3]
    V = R.shape[1]
    cam = torch.einsum("bvik,bvjk->bvji", R, points.unsqueeze(1).expand(-1, V, -1, -1)) + t.unsqueeze(2)
    x, y, z = cam[..., 0], cam[..., 1], cam[..., 2].clamp(min=1e-3)
    fx, fy = intrinsics[:, :, 0, 0].unsqueeze(-1), intrinsics[:, :, 1, 1].unsqueeze(-1)
    cx, cy = intrinsics[:, :, 0, 2].unsqueeze(-1), intrinsics[:, :, 1, 2].unsqueeze(-1)
    pixels = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    zero = torch.zeros_like(x)
    d_pixel_d_cam = torch.stack([torch.stack([fx / z, zero, -fx * x / z ** 2], dim=-1),
                                 torch.stack([zero, fy / z, -fy * y / z ** 2], dim=-1)], dim=-2)
    return pixels, torch.einsum("bvjik,bvkl->bvjil", d_pixel_d_cam, R)


def sample_field(pixels: Tensor, field: Tensor, with_gradient: bool = False):
    """Bilinear sample of a per-joint field at per-joint pixels (clamped to the image).

    pixels (B,V,J,2); field (B,V,h,w,J,C) -> values (B,V,J,C) and, if ``with_gradient``, the spatial
    gradient d value / d (u,v) as (B,V,J,C,2).
    """
    B, V, J = pixels.shape[:3]
    h, w, C = field.shape[2], field.shape[3], field.shape[-1]
    u = pixels[..., 0].clamp(0.0, w - 1.0)
    v = pixels[..., 1].clamp(0.0, h - 1.0)
    u0 = u.detach().floor().long().clamp(0, w - 2)
    v0 = v.detach().floor().long().clamp(0, h - 2)
    wu = (u - u0.float()).unsqueeze(-1)
    wv = (v - v0.float()).unsqueeze(-1)
    b = torch.arange(B, device=pixels.device).view(B, 1, 1).expand(B, V, J).reshape(-1)
    s = torch.arange(V, device=pixels.device).view(1, V, 1).expand(B, V, J).reshape(-1)
    j = torch.arange(J, device=pixels.device).view(1, 1, J).expand(B, V, J).reshape(-1)

    def at(du: int, dv: int) -> Tensor:
        return field[b, s, (v0 + dv).reshape(-1), (u0 + du).reshape(-1), j, :].reshape(B, V, J, C)

    f00, f10, f01, f11 = at(0, 0), at(1, 0), at(0, 1), at(1, 1)
    value = f00 * (1 - wu) * (1 - wv) + f10 * wu * (1 - wv) + f01 * (1 - wu) * wv + f11 * wu * wv
    if not with_gradient:
        return value
    d_du = (f10 - f00) * (1 - wv) + (f11 - f01) * wv
    d_dv = (f01 - f00) * (1 - wu) + (f11 - f10) * wu
    return value, torch.stack([d_du, d_dv], dim=-1)


def precision_from_cholesky(chol: Tensor) -> Tensor:
    """Raw Cholesky parameters (...,3) -> 2x2 SPD precision (...,2,2), Omega = M M^T, M = [[l11,0],[l21,l22]],
    l11, l22 = softplus(.) + 1e-3."""
    l11 = F.softplus(chol[..., 0]) + 1e-3
    l21 = chol[..., 1]
    l22 = F.softplus(chol[..., 2]) + 1e-3
    o11, o12, o22 = l11 * l11, l11 * l21, l21 * l21 + l22 * l22
    return torch.stack([torch.stack([o11, o12], -1), torch.stack([o12, o22], -1)], -2)


# ---------------------------------------------------------------------------------------------- energies
class VoteFieldEnergy(Energy):
    """Uncertainty-weighted, Huber-robust reprojection onto the network's per-view joint vote fields."""

    name = "uafit_vote_field"

    def __init__(self, views: SceneViews, huber_radius_px: float):
        self.views = views
        self.huber_radius_px = huber_radius_px

    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        views = self.views
        joints, d_joints = state.landmarks, state.landmark_jac                         # (B,J,3), (B,J,3,P)
        B, J, _, P = d_joints.shape
        V = views.view_mask.shape[1]
        pixels, d_pixels = project_with_jacobian(joints, views.intrinsics, views.world_to_camera)
        residual, d_residual_d_pixel = sample_field(pixels, views.vote_field, with_gradient=True)
        d_residual_d_world = torch.einsum("bvjci,bvjik->bvjck", d_residual_d_pixel, d_pixels)          # (B,V,J,2,3)
        jacobian = torch.einsum("bvjck,bjkp->bvjcp", d_residual_d_world, d_joints).reshape(B, V * J, 2, P)

        precision = precision_from_cholesky(sample_field(pixels.detach(), views.precision_chol))        # (B,V,J,2,2)
        norm = residual.norm(dim=-1)
        huber = torch.where(norm <= self.huber_radius_px, torch.ones_like(norm),
                            torch.sqrt(self.huber_radius_px / norm.clamp(min=1e-9)))              # IRLS weight
        precision = precision * (huber * views.view_mask.view(B, V, 1))[..., None, None]
        return gauss_newton_block(residual.reshape(B, V * J, 2), jacobian, precision.reshape(B, V * J, 2, 2))


# ---------------------------------------------------------------------------------------------- solve
@dataclass
class HandFit:
    """Result of one batched fit."""
    params: Tensor          # (B,P) hand-model parameters
    joints: Tensor          # (B,21,3) MANO-convention joints, metres, world
    mano_vertices: Tensor   # (B,778,3) MANO-topology vertices, metres, world
    final_cost: Tensor      # (B,)


def fit_hands(hand: FittableHand, evidence, huber_radius_px: float, settings: FitSettings = FitSettings()) -> HandFit:
    """Fit ``hand`` to a batch of ``NetworkEvidence`` (one hand per scene)."""
    views = SceneViews.from_network(evidence)
    triangulated = evidence.joints_world.float()
    hand.set_shape(evidence.mano_betas.float().detach())
    init = hand.initial_params(triangulated)
    energies = [
        VoteFieldEnergy(views, huber_radius_px * settings.huber_radius_scale),
        Anchor3DEnergy(triangulated, weight=settings.joint_anchor_weight),
        *hand.prior_energies(init),
    ]
    solver = LMSolver(max_iters=settings.iterations, damping=settings.lm_damping, solve_dtype=None)
    with torch.no_grad():
        result = solver.solve(hand.model, init, energies)
        return HandFit(params=result.params, joints=hand.landmarks(result.params),
                       mano_vertices=hand.mano_vertices(result.params),
                       final_cost=result.diagnostics["final_cost"])
