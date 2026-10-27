"""Phase A of the released model: precision-weighted dense-vote consensus.

Every pixel of the heatmap votes for a joint's 2D location
(``vote = pixel + offset``), each with an anisotropic per-vote precision
(from the Cholesky field). The consensus 2D point is the precision-weighted
MLE of the votes, and the fused per-view 2D precision falls out of the same
normal matrix. Parameter-free except for a single learned scalar
temperature ``log_tau``.

Dropped vs. the legacy ``lib/models/dovf/epipolar_filter.py::precision_consensus_2d``:
the ``ess`` (effective-sample-size) temperature-scaling branch -- the
released eval config never sets ``EPI.ESS_TAU`` (default ``False``), so the
branch is dead for this checkpoint.
"""
import torch


def chol_to_prec(raw: torch.Tensor) -> torch.Tensor:
    """(..., 3) raw Cholesky params -> (..., 2, 2) SPD precision Omega = M @ M.T.

    M = [[l11, 0], [l21, l22]], l11/l22 > 0 via softplus (+1e-3 floor).
    Information form: larger Omega = more confident.
    """
    eps = 1e-3
    l11 = torch.nn.functional.softplus(raw[..., 0]) + eps
    l21 = raw[..., 1]
    l22 = torch.nn.functional.softplus(raw[..., 2]) + eps
    o11 = l11 * l11
    o12 = l11 * l21
    o22 = l21 * l21 + l22 * l22
    row0 = torch.stack([o11, o12], dim=-1)
    row1 = torch.stack([o12, o22], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _pixel_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)                               # (h, w, 2)


def precision_consensus_2d(hm_probs: torch.Tensor, vote_field: torch.Tensor,
                           vote_chol: torch.Tensor, log_tau: torch.Tensor):
    """Precision-weighted MLE consensus of the dense per-pixel votes.

    Args:
        hm_probs:   (BN, J, h, w) softmax heatmap, support weight per pixel.
        vote_field: (BN, J, 2, h, w) per-pixel offset; vote = pixel + offset.
        vote_chol:  (BN, J, 3, h, w) per-vote raw Cholesky params.
        log_tau:    scalar learned log-temperature; fused precision = A * exp(log_tau).

    Returns:
        cons   (BN, J, 2)    consensus 2D point, heatmap px.
        omega2 (BN, J, 2, 2) fused per-view 2D precision.
    """
    bn, j, h, w = hm_probs.shape
    grid = _pixel_grid(h, w, hm_probs.device, hm_probs.dtype)          # (h, w, 2)
    offsets = vote_field.permute(0, 1, 3, 4, 2)                        # (BN, J, h, w, 2)
    votes = grid.view(1, 1, h, w, 2) + offsets
    chol = vote_chol.permute(0, 1, 3, 4, 2)                            # (BN, J, h, w, 3)
    lam = chol_to_prec(chol)                                           # (BN, J, h, w, 2, 2)
    wt = hm_probs

    A = (lam * wt[..., None, None]).sum(dim=(2, 3))                    # (BN, J, 2, 2)
    lam_v = torch.einsum("bjhwac,bjhwc->bjhwa", lam, votes)
    b = (lam_v * wt[..., None]).sum(dim=(2, 3))                        # (BN, J, 2)
    eye2 = torch.eye(2, device=hm_probs.device, dtype=hm_probs.dtype)
    cons = torch.linalg.solve(A + 1e-4 * eye2, b.unsqueeze(-1)).squeeze(-1)

    omega2 = A * torch.exp(log_tau)
    return cons, omega2
