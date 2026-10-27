"""Calibrate where MANO's 21 joints sit on another hand model (MHR or UmeTrack).

UA-Fit's network predicts MANO-convention joints. To fit MHR or UmeTrack to the
same network evidence, every MANO joint is attached to one bone of the target
model with a fixed offset (plus a dense MANO-vertex -> target-triangle map for
vertex metrics). This script calibrates that correspondence on synthetic MANO
hands and writes ``assets/correspondences/<model>_right.npz``. See
``parafit.mapping.calibration`` for the algorithm.

    python scripts/build_mano_correspondence.py --model mhr --mhr-cache assets/mhr/mhr_lod1_cache.pt
    python scripts/build_mano_correspondence.py --model umetrack --umetrack-json assets/umetrack/generic_hand_model.json
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uafit.hand_models import load_articulated_hand  # noqa: E402
from parafit.mapping.calibration import calibrate_mano_correspondence  # noqa: E402
from parafit.models.mano import ManoModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=["mhr", "umetrack"])
    p.add_argument("--mano-assets", default="assets/mano_v1_2")
    p.add_argument("--mhr-cache", default="assets/mhr/mhr_lod1_cache.pt")
    p.add_argument("--umetrack-json", default="assets/umetrack/generic_hand_model.json")
    p.add_argument("--out", default=None, help="default: assets/correspondences/<model>_right.npz")
    p.add_argument("--samples", type=int, default=512, help="synthetic MANO hands used for calibration")
    p.add_argument("--heldout", type=int, default=256, help="synthetic MANO hands used to report the residual")
    p.add_argument("--rounds", type=int, default=4, help="alternating fit / re-attach rounds")
    p.add_argument("--iterations", type=int, default=30, help="LM iterations per fit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    hand = load_articulated_hand(args.model, side="right", device=args.device, with_mesh=True,
                                 mhr_cache=args.mhr_cache, umetrack_json=args.umetrack_json)
    mano = ManoModel(args.mano_assets, side="right", center_idx=0, flat_hand_mean=True)
    correspondence = calibrate_mano_correspondence(
        hand, mano, args.mano_assets, f"{args.model}_right", n_fit=args.samples, n_val=args.heldout,
        rounds=args.rounds, fit_iterations=args.iterations, seed=args.seed)
    out = args.out or f"assets/correspondences/{args.model}_right.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    correspondence.save(out)
    print(f"wrote {out}")
    print("calibration:", correspondence.calibration)


if __name__ == "__main__":
    main()
