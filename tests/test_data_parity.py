"""Parity gate: `uafit.data.build_test_loader` vs the legacy eval pipeline.

For each of Table 1's 5 datasets, builds the legacy loader exactly the way
`scripts/eval.py::eval_dataset` does (`create_dataset` + `wds.WebLoader` +
`collation_random_n_views`, same `VIEW_RANGE`/`RECROP_FROM_JOINTS` overrides,
seed 1) and the new `uafit.data.build_test_loader`, then asserts every key
the network/`eval.py` actually consume is identical for the first 3 batches
at `workers=0`, plus a `workers=4` / 200-scene `sample_idx`-set check.

Run on risotto (has the shard data), from the repo root, with the `hoienv`
conda env:
    /opt/conda/envs/hoienv/bin/python tests/test_data_parity.py
"""
import gc
import os
import sys
import time

import numpy as np
import torch
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.datasets import create_dataset
from lib.utils.collation import collation_random_n_views
from lib.utils.config import get_config
from lib.utils.net_utils import setup_seed, worker_init_fn

from uafit.data import TEST_SPLITS, build_test_loader

DATA_ROOT = "data/dataset_tars"
CONFIG = "configs/uafit_eval.yaml"

# dataset key -> (config test-section, fixed max views) -- same mapping as
# `scripts/eval.py::DATASETS` (full-split sizes are cross-checked against
# `uafit.data.TEST_SPLITS` separately, see `check_full_sizes`).
DATASETS = {
    "DexYCB": ("DEXYCB_TEST", 8),
    "HO3D": ("HO3D_TEST", 5),
    "ARCTIC": ("ARCTIC_TEST", 8),
    "InterHand": ("INTERHAND_TEST", 8),
    "OakInk": ("OAKINK_TEST", 4),
}

# Keys the network (`uafit.network` / legacy `lib.models.dovf_mano_mv*`) and
# `scripts/eval.py` actually read out of a batch. `target_mano_shape` is
# optional (guarded by `"target_mano_shape" in batch` everywhere it's read)
# but cheap to carry and is checked too.
CONSUMED_KEYS = [
    "image",
    "target_cam_intr",
    "target_cam_extr",
    "cam_view_num",
    "master_joints_3d",
    "master_verts_3d",
    "target_mano_shape",
]


def build_legacy_cfg():
    cfg = get_config(CONFIG, arg=None, merge=True)
    cfg.defrost()
    if "OAKINK_TEST" not in cfg.DATASET.TEST:
        ok = cfg.DATASET.TEST.ARCTIC_TEST.clone()
        ok.defrost()
        ok.DATA_SPLIT = "test"
        ok.URLS = "data/dataset_tars/Oakink_mv/Oakink_mv_test-{000000..000045}.tar"
        cfg.DATASET.TEST.OAKINK_TEST = ok
    return cfg


def build_legacy_loader(cfg, section, nv, limit, batch_size, workers, seed):
    """Mirror `scripts/eval.py::eval_dataset`'s data construction exactly
    (minus the model), reseeding immediately before `create_dataset`, as
    `eval.py::main` does via `setup_seed(1)`."""
    ce = cfg.DATASET.TEST.clone()
    ce.defrost()
    ce.DATASET_LIST = [section]
    ce.EPOCH_SIZE = limit
    s = getattr(ce, section)
    s.VIEW_RANGE = [nv, nv]
    s.EPOCH_SIZE = limit
    if section == "INTERHAND_TEST":
        s.RECROP_FROM_JOINTS = True
    try:
        s.MIX_RATIO = 1.0
    except Exception:
        pass

    setup_seed(seed)
    de = create_dataset(ce, data_preset=cfg.DATA_PRESET, is_train=False)
    return wds.WebLoader(
        de, batch_size=batch_size, num_workers=workers, worker_init_fn=worker_init_fn,
        collate_fn=collation_random_n_views,
    )


def take_batches(loader, n):
    """Collect the first `n` batches, then drop the (possibly
    not-yet-exhausted) `DataLoader` iterator and force a GC pass.

    With `workers > 0`, breaking out of a partially-consumed
    `DataLoader` leaves its worker processes (and the queues/pipes
    referencing them) alive until Python collects the abandoned
    `_MultiProcessingDataLoaderIter` -- which, left to normal refcounting,
    can happen *later*, interleaved with whatever loader this process
    builds next. Observed effect: building+partially-iterating the legacy
    loader, immediately followed by building+iterating `uafit`'s loader
    (both seeded identically) made `uafit`'s loader deviate -- an artifact
    of two DataLoaders' worker pools being alive/torn-down concurrently,
    not a bug in `uafit.data` (confirmed: `build_test_loader`'s own output
    is identical whether or not a legacy loader was ever constructed
    first, as long as its worker pool is torn down before the next
    `DataLoader` is iterated). Forcing the GC here makes that explicit
    instead of relying on incidental timing.
    """
    out = []
    for i, b in enumerate(loader):
        out.append(b)
        if i + 1 >= n:
            break
    del loader
    gc.collect()
    return out


def reshape_legacy(batch, B):
    """Legacy's generic per-key `np.concatenate` collapses the per-SCENE
    (not per-view) `master_joints_3d`/`master_verts_3d` from `(B, J, 3)` into
    `(B*J, 3)` before the model's own `.view(B, J, 3)` puts them back. Redo
    that reshape here so both sides compare at the same shape; everything
    else in `CONSUMED_KEYS` is genuinely per-view in both pipelines and
    needs no such massaging."""
    mj = batch["master_joints_3d"].view(B, 21, 3)
    mv = batch["master_verts_3d"].view(B, 778, 3)
    return mj, mv


def compare_batch(name, legacy, ours, batch_idx):
    errs = []
    B = legacy["cam_view_num"].shape[0] if torch.is_tensor(legacy["cam_view_num"]) else len(legacy["cam_view_num"])
    l_mj, l_mv = reshape_legacy(legacy, B)
    o_mj, o_mv = ours["master_joints_3d"], ours["master_verts_3d"]

    def cmp_tensor(key, l, o):
        l = l.detach().cpu().float()
        o = o.detach().cpu().float() if torch.is_tensor(o) else torch.as_tensor(np.asarray(o)).float()
        if l.shape != o.shape:
            errs.append(f"{key}: shape mismatch legacy={tuple(l.shape)} ours={tuple(o.shape)}")
            return
        d = (l - o).abs().max().item()
        if d != 0.0:
            errs.append(f"{key}: max abs diff = {d:g}")

    cmp_tensor("image", legacy["image"], ours["image"])
    cmp_tensor("target_cam_intr", legacy["target_cam_intr"], ours["target_cam_intr"])
    cmp_tensor("target_cam_extr", legacy["target_cam_extr"], ours["target_cam_extr"])
    cmp_tensor("master_joints_3d", l_mj, o_mj)
    cmp_tensor("master_verts_3d", l_mv, o_mv)
    cmp_tensor("target_mano_shape", legacy["target_mano_shape"], ours["target_mano_shape"])

    # cam_view_num: plain numpy on both sides.
    l_cvn = np.asarray(legacy["cam_view_num"])
    o_cvn = np.asarray(ours["cam_view_num"])
    if not np.array_equal(l_cvn, o_cvn):
        errs.append(f"cam_view_num: legacy={l_cvn.tolist()} ours={o_cvn.tolist()}")

    # sample_idx: legacy's generic collation blindly routes every ndarray
    # key through `torch.Tensor(...)`, which always casts to float32 -- so
    # legacy's "sample_idx" is an *integer-valued float32* tensor (eval.py
    # itself calls `.long()` on it before using it as an index/set key, see
    # `scripts/eval.py::eval_dataset`). Normalise both sides to int64 before
    # comparing; this is a dtype cleanup, not a value difference.
    l_sid = legacy["sample_idx"].long().cpu().numpy() if torch.is_tensor(legacy["sample_idx"]) else np.asarray(legacy["sample_idx"]).astype(np.int64)
    o_sid = ours["sample_idx"].long().cpu().numpy() if torch.is_tensor(ours["sample_idx"]) else np.asarray(ours["sample_idx"]).astype(np.int64)
    if not np.array_equal(l_sid, o_sid):
        errs.append(f"sample_idx: legacy={l_sid.tolist()} ours={o_sid.tolist()}")

    if errs:
        print(f"  [FAIL] {name} batch {batch_idx}:")
        for e in errs:
            print(f"      {e}")
        return False
    print(f"  [OK]   {name} batch {batch_idx}: B={B}, views={l_cvn.tolist()}")
    return True


def run_workers0_check(name, section, nv, n_batches=3, batch_size=4, seed=1):
    cfg = build_legacy_cfg()
    legacy_loader = build_legacy_loader(cfg, section, nv, limit=0, batch_size=batch_size, workers=0, seed=seed)
    legacy_batches = take_batches(legacy_loader, n_batches)
    del legacy_loader
    gc.collect()

    ours_loader = build_test_loader(name, data_root=DATA_ROOT, limit=0, batch_size=batch_size, workers=0, seed=seed)
    ours_batches = take_batches(ours_loader, n_batches)
    del ours_loader
    gc.collect()

    if len(legacy_batches) != len(ours_batches):
        print(f"  [FAIL] {name}: got {len(legacy_batches)} legacy batches vs {len(ours_batches)} ours")
        return False

    ok = True
    for i, (lb, ob) in enumerate(zip(legacy_batches, ours_batches)):
        ok &= compare_batch(name, lb, ob, i)
    return ok


def collect_sample_idx(loader, n_scenes):
    """Scene-identity proxy: the `sample_idx` of each scene's *master* view
    (same convention `scripts/eval.py::eval_dataset` uses for its own
    dedup). Drops the loader (see `take_batches`'s docstring on why) once
    done -- important here too, since this is also called back-to-back for
    two different loaders."""
    seen = set()
    for b in loader:
        sid = b["sample_idx"]
        sid = sid.long().cpu().numpy() if torch.is_tensor(sid) else np.asarray(sid).astype(np.int64)
        cvn = np.asarray(b["cam_view_num"])
        first = np.concatenate([[0], np.cumsum(cvn)[:-1]])
        seen.update(int(x) for x in sid[first])
        if len(seen) >= n_scenes:
            break
    del loader
    gc.collect()
    return seen


def run_workers4_set_check(name, section, nv, n_scenes=200, batch_size=4, workers=4, seed=1):
    cfg = build_legacy_cfg()
    legacy_loader = build_legacy_loader(cfg, section, nv, limit=0, batch_size=batch_size, workers=workers, seed=seed)
    legacy_set = collect_sample_idx(legacy_loader, n_scenes)

    ours_loader = build_test_loader(name, data_root=DATA_ROOT, limit=0, batch_size=batch_size, workers=workers, seed=seed)
    ours_set = collect_sample_idx(ours_loader, n_scenes)
    legacy_200 = set(list(legacy_set)[:n_scenes])
    ours_200 = set(list(ours_set)[:n_scenes])
    ok = legacy_200 == ours_200
    print(f"  [{'OK' if ok else 'FAIL'}] {name} workers=4 sample_idx set over ~200 scenes: "
          f"legacy n={len(legacy_200)} ours n={len(ours_200)} "
          f"intersection={len(legacy_200 & ours_200)}")
    return ok


def check_full_sizes():
    from scripts.eval import DATASETS as EVAL_DATASETS

    ok = True
    for name, (_, _, full_n) in EVAL_DATASETS.items():
        reg_n = TEST_SPLITS[name].full_size
        if reg_n != full_n:
            print(f"  [FAIL] {name}: TEST_SPLITS.full_size={reg_n} != scripts/eval.py DATASETS full_n={full_n}")
            ok = False
    if ok:
        print("  [OK]   TEST_SPLITS.full_size matches scripts/eval.py::DATASETS for all 5 datasets")
    return ok


def main():
    overall = True
    print("== full-split-size registry cross-check ==")
    overall &= check_full_sizes()

    print("\n== workers=0, first-3-batches bit-exact parity ==")
    for name, (section, nv) in DATASETS.items():
        t0 = time.time()
        ok = run_workers0_check(name, section, nv)
        overall &= ok
        print(f"  ({name} done in {time.time() - t0:.1f}s)")

    print("\n== workers=4, first ~200 scenes sample_idx SET parity ==")
    for name, (section, nv) in DATASETS.items():
        t0 = time.time()
        ok = run_workers4_set_check(name, section, nv)
        overall &= ok
        print(f"  ({name} done in {time.time() - t0:.1f}s)")

    print("\n" + ("ALL PARITY CHECKS PASSED" if overall else "PARITY CHECKS FAILED"))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
