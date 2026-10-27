"""Numerical parity gate: uafit.network.UAFitNetwork vs. the legacy
DOVFManoMVEpi forward it replaces.

Run on risotto (GPU, manotorch, dataset shards are all there):

    cd /code/uafit-release/ua-fit-code
    /opt/conda/envs/hoienv/bin/python tests/test_network_parity.py --device cuda:0

Builds BOTH models from the SAME released checkpoint
(``checkpoints/uafit_w40``), runs 2 batches each (batch_size=4, workers=0)
from the HO3D and DexYCB test shards through:

  legacy:  p = model._forward_impl(batch, mode="val")
           -> p["mu3d"], p["beta_hat"], p["dovf_field"], p["chol_field"]
  network: ev = network(batch)
           -> ev.joints_world, ev.mano_betas, ev.vote_field, ev.vote_precision_chol

and asserts max-abs-diff <= 1e-5 for every field (plus the derived
intrinsics_heatmap / world_to_camera / views_per_scene geometry).

Also reports the checkpoint keys UAFitNetwork does not load (training-only /
disabled-at-eval modules) and asserts each is one of the known-dead groups
documented in ``uafit/network/network.py``'s module docstring.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import eval as legacy_eval  # scripts/eval.py -- READ ONLY, never modified
from lib.utils.collation import collation_random_n_views
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.datasets import create_dataset
import webdataset as wds

from uafit.network import load_uafit_network

# Checkpoint-key groups NOT loaded into UAFitNetwork, and why. Any unexpected
# key outside these groups fails the test (it would mean an eval-path module
# was silently dropped).
KNOWN_IGNORED_PREFIXES = {
    "descriptor_head.": "DESC_DIM=64 trained, but output unused by every NetworkEvidence field "
                        "(only feeds the train-time correspondence loss / MESH_IMG, which is off)",
    "pose_refiner.": "EPI.POSE_REFINE trained True, forced False by scripts/eval.py::build_model "
                     "for the paper's efficiency-inference numbers -> module not built at eval",
    "refiner.": "EpipolarRefiner: EPI.T=0 in configs/uafit_eval.yaml makes the Phase-C refine loop "
               "run zero iterations (_scene_pair_schedule returns an empty step list) -> "
               "refiner.forward() is never called at eval time",
    "prior_": "solver pose prior (PCA-Mahalanobis); rebuilt from MANO's PCA in uafit.hand_models.ManoHand, "
              "identical values (checked in tests/test_fitter_parity.py)",
    "mano_layer.": "redundant clone of the deterministic assets/mano_v1_2 buffers; "
                  "the solver builds MANO from the same assets (uafit.hand_models)",
    "imagenet_mean": "replaced by an equivalent hardcoded constant in UAFitNetwork._renorm",
    "imagenet_std": "replaced by an equivalent hardcoded constant in UAFitNetwork._renorm",
    "J_regressor": "top-level duplicate of mano_layer.th_J_regressor (same values)",
    "_beta_std": "EMA buffer for the train-time standardized beta loss only",
    "img_backbone.classifier": "HRNet ImageNet classification head; HighResolutionNet.forward() "
                               "never calls it even in the legacy model",
    "img_backbone.incre_modules": "HRNet ImageNet classification head (see above)",
    "img_backbone.downsamp_modules": "HRNet ImageNet classification head (see above)",
    "img_backbone.final_layer": "HRNet ImageNet classification head (see above)",
}


def classify_ignored_key(key: str) -> str:
    for prefix, reason in KNOWN_IGNORED_PREFIXES.items():
        if key.startswith(prefix) or key == prefix:
            return reason
    return None


def make_loader(cfg, section: str, nv: int, epoch_size: int, batch_size: int):
    ce = cfg.DATASET.TEST.clone()
    ce.defrost()
    ce.DATASET_LIST = [section]
    ce.EPOCH_SIZE = epoch_size
    s = getattr(ce, section)
    s.VIEW_RANGE = [nv, nv]
    s.EPOCH_SIZE = epoch_size
    try:
        s.MIX_RATIO = 1.0
    except Exception:
        pass
    de = create_dataset(ce, data_preset=cfg.DATA_PRESET, is_train=False)
    return wds.WebLoader(de, batch_size=batch_size, num_workers=0,
                         worker_init_fn=worker_init_fn, collate_fn=collation_random_n_views)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def run_parity(device_str: str, n_batches: int = 2, batch_size: int = 4):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_str)
    setup_seed(1)

    args = SimpleNamespace(ckpt="checkpoints/uafit_w40", config="configs/uafit_eval.yaml",
                           backbone="HRNet", iters=15, gnscale=3e5, jac="analytic",
                           rad_mult=0.5, datasets=["HO3D", "DexYCB"], views=0, limit=8,
                           batch_size=batch_size, workers=0, device=device_str)
    cfg, legacy_model = legacy_eval.build_model(args, device)
    legacy_model.eval()

    network = load_uafit_network(args.ckpt, device=device)
    print(f"UAFitNetwork ignored checkpoint keys: {len(network.ignored_checkpoint_keys)}")
    unexplained = []
    for k in network.ignored_checkpoint_keys:
        reason = classify_ignored_key(k)
        if reason is None:
            unexplained.append(k)
    if unexplained:
        raise AssertionError(f"ignored checkpoint keys with no documented reason: {unexplained}")
    by_prefix = {}
    for k in network.ignored_checkpoint_keys:
        top = k.split(".")[0]
        by_prefix[top] = by_prefix.get(top, 0) + 1
    print("  ignored key groups ->", by_prefix)

    print(f"learned Huber radius = {network.huber_radius_px:.4f} heatmap px")

    specs = [("HO3D_TEST", 5), ("DEXYCB_TEST", 8)]
    worst = {}
    n_checked = 0
    for section, nv in specs:
        loader = make_loader(cfg, section, nv, epoch_size=batch_size * n_batches * 2, batch_size=batch_size)
        seen = 0
        for b in loader:
            b = legacy_eval.cuda_batch(b, device)
            if "master_verts_3d" not in b:
                continue
            with torch.no_grad():
                p = legacy_model._forward_impl(b, mode="val", epoch_idx=0)
                ev = network(b)

            k_all = b["target_cam_intr"].reshape(-1, 3, 3)
            sx, sy = legacy_model._hm_scale(device, k_all.dtype)
            k_hm_ref = k_all.clone()
            k_hm_ref[:, 0, :] *= sx
            k_hm_ref[:, 1, :] *= sy
            w2c_ref = torch.linalg.inv(b["target_cam_extr"].reshape(-1, 4, 4))

            diffs = {
                "joints_world (mu3d)": max_abs_diff(ev.joints_world, p["mu3d"]),
                "mano_betas (beta_hat)": max_abs_diff(ev.mano_betas, p["beta_hat"]),
                "vote_field (dovf_field)": max_abs_diff(ev.vote_field, p["dovf_field"]),
                "vote_precision_chol (chol_field)": max_abs_diff(ev.vote_precision_chol, p["chol_field"]),
                "intrinsics_heatmap": max_abs_diff(ev.intrinsics_heatmap, k_hm_ref),
                "world_to_camera": max_abs_diff(ev.world_to_camera, w2c_ref),
            }
            views_ref = [int(n) for n in b["cam_view_num"]]
            assert ev.views_per_scene == views_ref, (ev.views_per_scene, views_ref)
            assert ev.heatmap_size == tuple(legacy_model.heatmap_size)

            print(f"[{section} batch {seen}] " +
                 " | ".join(f"{k}: {v:.3e}" for k, v in diffs.items()))
            for k, v in diffs.items():
                worst[k] = max(worst.get(k, 0.0), v)
            n_checked += 1
            seen += 1
            if seen >= n_batches:
                break

    print("\n=== worst-case max-abs-diff across all checked batches ===")
    ok = True
    for k, v in worst.items():
        status = "OK" if v <= 1e-5 else "FAIL"
        if v > 1e-5:
            ok = False
        print(f"  {k:36s} {v:.3e}  [{status}]")
    assert n_checked >= 2, f"only checked {n_checked} batches; expected >= 2 (HO3D + DexYCB)"
    assert ok, "parity gate FAILED: some field exceeded 1e-5 max-abs-diff"
    print(f"\nPARITY GATE PASSED ({n_checked} batches checked)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    run_parity(a.device, n_batches=a.n_batches, batch_size=a.batch_size)
