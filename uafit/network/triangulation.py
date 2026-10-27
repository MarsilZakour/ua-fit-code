"""Phase B of the released model: parameter-free, uncertainty-weighted
multi-view triangulation of the per-view 2D consensus into a 3D joint
estimate ("mu3d" / ``joints_world``).

Ported from ``lib/models/dovf/triangulate_resample.py``; the MixtureMCHead /
CostVolumeHead learned scorers in that file are a different (unused, never
built for the released checkpoint -- ``UNC.MC_HEAD=False``) code path and are
not reproduced here.

Camera convention: ``w2c = [R|t]`` world->cam; 2D points and intrinsics are
in HEATMAP pixels (see :func:`uafit.network.network.UAFitNetwork._hm_scale`).
"""
import torch


def pad_by_scene(x: torch.Tensor, views_per_scene, n_max: int):
    """(sum(views_per_scene), ...) -> (B, n_max, ...) zero-padded + (B, n_max) mask."""
    b = len(views_per_scene)
    out = x.new_zeros(b, n_max, *x.shape[1:])
    vmask = torch.zeros(b, n_max, device=x.device)
    for i in range(b):
        s = int(sum(views_per_scene[:i])); e = int(sum(views_per_scene[:i + 1])); n = e - s
        out[i, :n] = x[s:e]
        vmask[i, :n] = 1.0
    return out, vmask


def _unweighted_triangulate(cons_pad, k_pad, w2c_pad, vmask, eps=1e-3):
    """Unweighted multi-view ray intersection -> (mu, Sigma3), used only to
    seed :func:`triangulate_omega2`'s Gauss-Newton iterations."""
    b, n, j, _ = cons_pad.shape
    r = w2c_pad[..., :3, :3]; t = w2c_pad[..., :3, 3]
    rt = r.transpose(-1, -2)
    cam_c = -torch.einsum("bnij,bnj->bni", rt, t)                      # (B, N, 3)
    uvh = torch.cat([cons_pad, torch.ones_like(cons_pad[..., :1])], -1)
    eye3 = torch.eye(3, device=k_pad.device, dtype=k_pad.dtype)
    k_safe = torch.where(vmask[..., None, None] > 0, k_pad, eye3)      # avoid inv() on padded zeros
    d = torch.einsum("bnij,bnkj->bnki", rt,
                     torch.einsum("bnij,bnkj->bnki", torch.linalg.inv(k_safe), uvh))
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    m = torch.eye(3, device=d.device, dtype=d.dtype) - d.unsqueeze(-1) * d.unsqueeze(-2)  # (B,N,J,3,3)
    eye = torch.eye(3, device=d.device, dtype=d.dtype)
    vm = vmask.view(b, n, 1, 1, 1)
    a = (vm * m).sum(1) + eps * eye                                    # (B, J, 3, 3)
    mc = torch.einsum("bnjac,bnc->bnja", m, cam_c)
    bvec = (vmask.view(b, n, 1, 1) * mc).sum(1)
    sigma = torch.linalg.inv(a)
    mu = torch.einsum("bjac,bjc->bja", sigma, bvec)
    return mu, sigma


def triangulate_omega2(cons_pad: torch.Tensor, k_pad: torch.Tensor, w2c_pad: torch.Tensor,
                       omega2_pad: torch.Tensor, vmask: torch.Tensor, n_iter: int = 2,
                       robust_iter: int = 4, robust_c2: float = 25.0, eps: float = 1e-4):
    """Robust (IRLS), uncertainty-weighted Gauss-Newton triangulation.

    Minimizes sum_v r_v^T Omega2_v r_v over the reprojection residual. After
    ``n_iter`` plain GN steps, runs ``robust_iter`` additional IRLS steps that
    reweight each (view, joint) by a Cauchy/Lorentzian function of its
    Mahalanobis reprojection residual -- this is what lets DOVFManoMVEpi
    reject a wrong-hand / fully-occluded view (e.g. InterHand's two-hand
    ambiguity) at inference with zero extra parameters.

    This checkpoint's eval config (``scripts/eval.py::build_model`` forces
    ``EPI.ROBUST_TRIANG=True``) always takes the robust branch with
    ``ROBUST_KIND="cauchy"`` (the default); the alternate "huber"/"master"
    weighting kinds in the legacy code are dead for this checkpoint and are
    not reproduced here.

    Args:
        cons_pad:   (B, N, J, 2) per-view 2D consensus, heatmap px.
        k_pad:      (B, N, 3, 3) heatmap-pixel intrinsics.
        w2c_pad:    (B, N, 4, 4) world->cam.
        omega2_pad: (B, N, J, 2, 2) per-(view,joint) 2D precision.
        vmask:      (B, N) 1.0 = real view, 0.0 = padding.

    Returns:
        mu     (B, J, 3)    triangulated joints, world frame.
        sigma3 (B, J, 3, 3) triangulation covariance (Sigma3 = A^-1); not
                            required by :class:`NetworkEvidence` (the fitter
                            recomputes its own 3D-anchor precision) but
                            returned for completeness/debugging.
    """
    b, n, j, _ = cons_pad.shape
    r = w2c_pad[..., :3, :3]; t = w2c_pad[..., :3, 3]
    fx = k_pad[..., 0, 0, None]; fy = k_pad[..., 1, 1, None]
    cx = k_pad[..., 0, 2, None]; cy = k_pad[..., 1, 2, None]
    eye = torch.eye(3, device=cons_pad.device, dtype=cons_pad.dtype)
    mu, _ = _unweighted_triangulate(cons_pad, k_pad, w2c_pad, vmask)
    vm = vmask.view(b, n, 1, 1, 1)
    sigma3 = None
    for it in range(n_iter + robust_iter):
        xc = torch.einsum("bnij,bkj->bnki", r, mu) + t[:, :, None, :]  # (B, N, J, 3)
        x, y = xc[..., 0], xc[..., 1]
        z = xc[..., 2].clamp(min=1e-4)
        uvp = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)      # (B, N, J, 2)
        j_uvc = torch.zeros(b, n, j, 2, 3, device=cons_pad.device, dtype=cons_pad.dtype)
        j_uvc[..., 0, 0] = fx / z; j_uvc[..., 0, 2] = -fx * x / (z * z)
        j_uvc[..., 1, 1] = fy / z; j_uvc[..., 1, 2] = -fy * y / (z * z)
        jac = torch.einsum("bnjac,bncd->bnjad", j_uvc, r)              # d(uv)/d(X_world)
        resid = cons_pad - uvp                                         # (B, N, J, 2)

        om = omega2_pad
        if it >= n_iter:
            rho = torch.einsum("bnja,bnjac,bnjc->bnj", resid, omega2_pad, resid).clamp(min=0.0)
            wr = robust_c2 / (robust_c2 + rho)                         # Cauchy/Lorentzian downweight
            om = omega2_pad * wr[..., None, None]

        jt_om = torch.einsum("bnjca,bnjcd->bnjad", jac, om)            # J^T Omega
        h = torch.einsum("bnjad,bnjde->bnjae", jt_om, jac)             # (B, N, J, 3, 3)
        g = torch.einsum("bnjad,bnjd->bnja", jt_om, resid)             # (B, N, J, 3)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        a = (vm * h).sum(1)                                            # (B, J, 3, 3)
        tr_a = torch.diagonal(a, dim1=-2, dim2=-1).sum(-1).clamp(min=0.0)
        a = a + (eps + 1e-4 * tr_a[..., None, None] / 3.0) * eye       # scale-aware Tikhonov
        gg = (vmask.view(b, n, 1, 1) * g).sum(1)
        sigma3 = torch.linalg.inv(a)
        mu = mu + torch.einsum("bjac,bjc->bja", sigma3, gg)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    return mu, sigma3
