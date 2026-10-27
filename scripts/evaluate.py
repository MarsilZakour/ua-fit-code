"""Reproduce UA-Fit Table 1, and fit MHR / UmeTrack hands to the same network evidence.

    # Table 1 (MANO, all five datasets, full official splits)
    python scripts/evaluate.py --checkpoint Marsil123/ua-fit

    # one dataset, quick subset
    python scripts/evaluate.py --checkpoint Marsil123/ua-fit --datasets HO3D --limit 200

    # the same checkpoint, fitting MHR or UmeTrack instead of MANO
    python scripts/evaluate.py --checkpoint Marsil123/ua-fit --hand-model mhr

Protocol (paper): full official split per dataset at a fixed view count (DexYCB 8,
HO3D 5, ARCTIC 8, InterHand2.6M 8, OakInk 4), seed 1, one fit per scene with the
feed-forward shape. Metrics in mm: MPVPE, MPJPE against mesh-regressed ground-truth
joints, PA-MPVPE. For MHR / UmeTrack the joints are the MANO-convention joints of the
fitted model and the vertices are MANO-topology points read off the fitted mesh
(see ``uafit.hand_models``), so all three metrics stay comparable to MANO ground truth.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uafit.data import TEST_SPLITS, batch_to_device, build_test_loader  # noqa: E402
from uafit.fitting import FitSettings, fit_hands  # noqa: E402
from uafit.hand_models import HAND_MODELS, HandModelPaths, build_hand  # noqa: E402
from uafit.metrics import mano_joints_from_vertices, mean_point_error, procrustes_aligned_error  # noqa: E402
from uafit.network import load_uafit_network  # noqa: E402

PAPER_TABLE1 = {  # MPVPE / MPJPE / PA-MPVPE (mm), UA-Fit (MANO)
    "DexYCB": (7.86, 7.55, 4.92), "HO3D": (8.01, 7.41, 5.10), "ARCTIC": (7.54, 6.62, 5.66),
    "InterHand": (8.77, 7.79, 6.23), "OakInk": (9.58, 8.86, 5.67),
}


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce UA-Fit Table 1.")
    p.add_argument("--checkpoint", default="Marsil123/ua-fit",
                   help="Hugging Face model id, a weights file, or a directory holding uafit_w40.safetensors")
    p.add_argument("--data-root", default="data/dataset_tars", help="directory holding the WebDataset shards")
    p.add_argument("--datasets", nargs="+", default=list(TEST_SPLITS), choices=list(TEST_SPLITS))
    p.add_argument("--hand-model", default="mano", choices=HAND_MODELS)
    p.add_argument("--views", type=int, default=0, help="cap the view count (0 = per-dataset protocol)")
    p.add_argument("--limit", type=int, default=0, help="scenes per dataset (0 = full official split)")
    p.add_argument("--iterations", type=int, default=FitSettings.iterations, help="LM iterations")
    p.add_argument("--joint-anchor-weight", type=float, default=FitSettings.joint_anchor_weight, help="lambda_3D")
    p.add_argument("--huber-radius-scale", type=float, default=FitSettings.huber_radius_scale)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="", help="optional JSON file for the results")
    return p.parse_args()


def evaluate_split(name, network, hand, settings, joint_regressor, args, device) -> dict:
    split = TEST_SPLITS[name]
    n_target = args.limit or split.full_size
    loader = build_test_loader(name, args.data_root, max_views=args.views or None, limit=n_target,
                               batch_size=args.batch_size, workers=args.workers, seed=args.seed)
    errors = {"mpvpe": [], "mpjpe": [], "pa_mpvpe": []}
    seen, scored = set(), 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.no_grad():
            evidence = network(batch)
            fit = fit_hands(hand, evidence, network.huber_radius_px, settings)
        gt_vertices = batch["master_verts_3d"].view(-1, 778, 3)
        gt_joints = mano_joints_from_vertices(gt_vertices, joint_regressor)
        valid = gt_vertices.norm(dim=-1).mean(-1) > 1e-4
        # score each scene once (the shard stream can repeat scenes at its end)
        first_view = np.concatenate([[0], np.cumsum(evidence.views_per_scene)[:-1]])
        scene_ids = batch["sample_idx"].view(-1)[torch.as_tensor(first_view, device=device)].long().tolist()
        fresh = torch.tensor([sid not in seen for sid in scene_ids], device=device)
        valid = valid & fresh
        seen.update(sid for sid, ok in zip(scene_ids, valid.tolist()) if ok)
        if valid.any():
            scored += int(valid.sum())
            errors["mpvpe"].append(mean_point_error(fit.mano_vertices[valid], gt_vertices[valid]).cpu())
            errors["mpjpe"].append(mean_point_error(fit.joints[valid], gt_joints[valid]).cpu())
            errors["pa_mpvpe"].append(procrustes_aligned_error(fit.mano_vertices[valid], gt_vertices[valid]).cpu())
        if scored >= n_target:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    result = {k: torch.cat(v).mean().item() for k, v in errors.items()}
    return dict(dataset=name, views=min(args.views or split.n_views, split.n_views), n=scored,
                fits_per_second=scored / max(seconds, 1e-6), **result)


def main():
    args = parse_args()
    device = torch.device(args.device)
    network = load_uafit_network(args.checkpoint, device=device)
    hand = build_hand(args.hand_model, HandModelPaths(), device=device)
    settings = FitSettings(iterations=args.iterations, joint_anchor_weight=args.joint_anchor_weight,
                           huber_radius_scale=args.huber_radius_scale)
    joint_regressor = build_hand("mano", device=device).model.J_regressor.to(device) if args.hand_model != "mano" \
        else hand.model.J_regressor.to(device)
    print(f"UA-Fit | hand model {args.hand_model} | {settings} | checkpoint {args.checkpoint}", flush=True)
    header = f"{'dataset':<11}{'views':>6}{'n':>7}{'MPVPE':>8}{'MPJPE':>8}{'PA-MPVPE':>10}{'fit/s':>8}"
    if args.hand_model == "mano":
        header += f"   paper (MPVPE / MPJPE / PA-MPVPE)"
    print(header, flush=True)
    rows = []
    for name in args.datasets:
        r = evaluate_split(name, network, hand, settings, joint_regressor, args, device)
        rows.append(r)
        line = (f"{name:<11}{r['views']:>6}{r['n']:>7}{r['mpvpe']:>8.2f}{r['mpjpe']:>8.2f}{r['pa_mpvpe']:>10.2f}"
                f"{r['fits_per_second']:>8.1f}")
        if args.hand_model == "mano" and not args.limit and not args.views:
            line += "   {:.2f} / {:.2f} / {:.2f}".format(*PAPER_TABLE1[name])
        print(line, flush=True)
    if len(rows) > 1:
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("mpvpe", "mpjpe", "pa_mpvpe")}
        print(f"{'mean':<11}{'':>13}{mean['mpvpe']:>8.2f}{mean['mpjpe']:>8.2f}{mean['pa_mpvpe']:>10.2f}", flush=True)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(dict(args=vars(args), settings=settings.__dict__, results=rows), f, indent=2)


if __name__ == "__main__":
    main()
