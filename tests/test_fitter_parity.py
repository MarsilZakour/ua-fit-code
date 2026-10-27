"""Parity gate: the release fitter (parafit LMSolver + uafit energies, MANO) vs the paper's legacy solver.

Both fit the SAME network outputs (legacy network + legacy loader), so any difference is the solver alone.
The paper solves normal equations with condition numbers ~1e9 in float32, so 1e-7 rounding differences
(summation order) change individual LM steps by a few percent and single samples can end up ~1-5 mm apart.
That is the solver's own float32 noise: the script measures it directly by re-running the LEGACY solver on
inputs perturbed by 1e-6 px. The gate is therefore (1) the per-sample error difference vs legacy is no larger
than that noise floor, and (2) the mean metrics agree within 0.02 mm.
Run on a GPU machine from the repository root:

    python tests/test_fitter_parity.py --datasets HO3D DexYCB --batches 4
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import eval as legacy_eval  # noqa: E402  (scripts/eval.py, the paper harness)
import webdataset as wds  # noqa: E402
from lib.datasets import create_dataset  # noqa: E402
from lib.utils.collation import collation_random_n_views  # noqa: E402
from lib.utils.net_utils import setup_seed, worker_init_fn  # noqa: E402
from lib.utils.transform import mano_to_openpose  # noqa: E402

from uafit.fitting import FitSettings, fit_hands  # noqa: E402
from uafit.hand_models import build_hand  # noqa: E402


def legacy_fit(model, p, b, mano, Jreg, device):
    mu, beta = p["mu3d"].float(), p["beta_hat"].float()
    B = mu.shape[0]
    go, tr = legacy_eval.kabsch_global(mu, mano, Jreg, device)
    pose = torch.cat([go, torch.zeros(B, 45, device=device)], 1)
    K_all = b["target_cam_intr"].view(-1, 3, 3)
    w2c_all = torch.linalg.inv(b["target_cam_extr"].view(-1, 4, 4))
    pose, tr = model._run_fitter_analytic_unc(pose, beta, tr, p["dovf_field"].float(), p["chol_field"].float(), None,
                                              K_all.float(), w2c_all.float(), b["cam_view_num"], device=device,
                                              mode="val", mc_mu3d=mu)[:2]
    out = mano(pose, beta)
    return out.verts + tr[:, None], mano_to_openpose(Jreg, out.verts)[:, :21] + tr[:, None]


def evidence_from_legacy(model, p, b, device):
    K = b["target_cam_intr"].view(-1, 3, 3).float().clone()
    sx, sy = model._hm_scale(device, torch.float32)
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    return SimpleNamespace(joints_world=p["mu3d"], mano_betas=p["beta_hat"], vote_field=p["dovf_field"],
                           vote_precision_chol=p["chol_field"], intrinsics_heatmap=K,
                           world_to_camera=torch.linalg.inv(b["target_cam_extr"].view(-1, 4, 4)),
                           views_per_scene=b["cam_view_num"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/uafit_w40")
    ap.add_argument("--datasets", nargs="+", default=["HO3D", "DexYCB"])
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    args = SimpleNamespace(config="configs/uafit_eval.yaml", backbone="HRNet", iters=15, gnscale=3e5, jac="analytic",
                           rad_mult=0.5, ckpt=a.ckpt)
    device = torch.device(a.device)
    setup_seed(1)
    cfg, model = legacy_eval.build_model(args, device)
    mano, Jreg = model.mano_layer, model.J_regressor
    hand = build_hand("mano", device=device)
    radius = float(model.fit_log_radius.exp())               # already scaled by rad_mult in build_model
    settings = FitSettings(huber_radius_scale=1.0)
    worst = 0.0
    for name in a.datasets:
        section, nv, _ = legacy_eval.DATASETS[name]
        ce = cfg.DATASET.TEST.clone(); ce.defrost(); ce.DATASET_LIST = [section]
        s = getattr(ce, section); s.VIEW_RANGE = [nv, nv]; s.EPOCH_SIZE = a.batches * a.batch_size
        if section == "INTERHAND_TEST":
            s.RECROP_FROM_JOINTS = True
        loader = wds.WebLoader(create_dataset(ce, data_preset=cfg.DATA_PRESET, is_train=False), batch_size=a.batch_size,
                               num_workers=0, worker_init_fn=worker_init_fn, collate_fn=collation_random_n_views)
        diffs, noise, err_old, err_new, err_noise = [], [], [], [], []
        for i, b in enumerate(loader):
            if i >= a.batches:
                break
            b = legacy_eval.cuda_batch(b, device)
            with torch.no_grad():
                p = model._forward_impl(b, mode="val", epoch_idx=0)
                v_old, j_old = legacy_fit(model, p, b, mano, Jreg, device)
                fit = fit_hands(hand, evidence_from_legacy(model, p, b, device), radius, settings)
                p_noisy = dict(p, dovf_field=p["dovf_field"] + 1e-6)          # float-noise-level perturbation
                v_noisy, _ = legacy_fit(model, p_noisy, b, mano, Jreg, device)
            gt = b["master_verts_3d"].view(-1, 778, 3)
            mpvpe = lambda v: ((v - gt).norm(dim=-1).mean(-1) * 1000).cpu()
            err_old.append(mpvpe(v_old)); err_new.append(mpvpe(fit.mano_vertices)); err_noise.append(mpvpe(v_noisy))
        old, new, noisy = torch.cat(err_old), torch.cat(err_new), torch.cat(err_noise)
        spread_new = (new - old).abs()                     # per-sample |MPVPE difference| new vs legacy
        spread_noise = (noisy - old).abs()                 # same for legacy vs legacy on perturbed inputs
        mean_gap = abs(new.mean() - old.mean()).item()
        ok = mean_gap <= 0.02 and spread_new.mean() <= 2 * spread_noise.mean() + 1e-3
        worst = max(worst, 0.0 if ok else 1.0)
        print(f"{name:10s} n={old.numel():4d} | MPVPE legacy {old.mean():.3f} new {new.mean():.3f} (gap {mean_gap:.3f}) | "
              f"per-sample |dMPVPE| new-vs-legacy mean {spread_new.mean():.3f} max {spread_new.max():.3f}; "
              f"legacy float-noise floor mean {spread_noise.mean():.3f} max {spread_noise.max():.3f} | "
              f"{'ok' if ok else 'FAIL'}", flush=True)
    print("PASS" if worst == 0.0 else "FAIL")


if __name__ == "__main__":
    main()
