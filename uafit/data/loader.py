"""Public entry point: `build_test_loader` + the `TEST_SPLITS` registry.

Why a Python registry instead of a YAML config: the legacy
`configs/uafit_eval.yaml` is ~330 lines because it carries the *training*
mix (6 datasets, augmentation knobs, mix ratios, curriculum) plus the test
sections. At eval time, only 5 facts per dataset actually vary: the shard
glob, the fixed view count, the official split size, the crop size (256 for
all 5, but kept explicit rather than hard-coded), and two boolean per-dataset
fixups (`RECROP_FROM_JOINTS`, extrinsics inversion). A 5-field dataclass
documents and type-checks that better than another small YAML file would,
with no loss of editability (this *is* the config).
"""
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .collate import collate_views
from .shards import group_views, open_shards
from .views import INV_EXTR_DATASETS, process_sample

_INT32_MOD = 2**32 - 1  # lib.utils.misc.CONST.INT_MAX


@dataclass(frozen=True)
class TestSplit:
    """One of Table 1's five held-out test/val splits."""

    url_pattern: str  # shard glob, relative to `data_root` (braceexpand-able)
    n_views: int  # fixed view count at eval (legacy VIEW_RANGE = [n_views, n_views])
    full_size: int  # number of scenes in the official split
    crop_size: int = 256  # output image side (DATA_PRESET.IMAGE_SIZE, square)
    recrop_from_joints: bool = False  # InterHand only: rebuild bbox from GT joints_2d
    invert_extrinsics: bool = False  # dataset name is in views.INV_EXTR_DATASETS


# Shard globs and official split sizes copied from `scripts/eval.py::DATASETS`
# and `configs/uafit_eval.yaml`'s `*_TEST` sections (OakInk's test section is
# synthesised in `eval.py::build_model` from the ARCTIC_TEST template).
TEST_SPLITS: Dict[str, TestSplit] = {
    "DexYCB": TestSplit("DexYCB_mv/DexYCB_mv_test-{000000..000003}.tar", n_views=8, full_size=4931),
    "HO3D": TestSplit("HO3D_mv_test/HO3D_mv_test-{000000..000002}.tar", n_views=5, full_size=2687),
    "ARCTIC": TestSplit(
        "Arctic_mv/Arctic_mv_val_p1-{000000..000045}.tar", n_views=8, full_size=17359, invert_extrinsics=True
    ),
    "InterHand": TestSplit(
        "Interhand_mv/Interhand_mv_val-{000000..000022}.tar",
        n_views=8,
        full_size=85235,
        recrop_from_joints=True,
        invert_extrinsics=True,
    ),
    "OakInk": TestSplit(
        "Oakink_mv/Oakink_mv_test-{000000..000045}.tar", n_views=4, full_size=21322, invert_extrinsics=True
    ),
}
# Sanity: TestSplit.invert_extrinsics is set per-dataset from
# views.INV_EXTR_DATASETS, keyed by the shard basename's first '_'-separated
# token (exactly how the legacy `MultiviewWebDataset.name` is derived) --
# NOT by this registry's (more readable) dict key, which differs in case
# from two of the five legacy-derived names ("InterHand" vs "Interhand",
# "OakInk" vs "Oakink"). Keeps the two tables in lockstep with any future edit.
def _legacy_derived_name(url_pattern: str) -> str:
    return url_pattern.split("/")[-1].split("_")[0]


assert all(
    spec.invert_extrinsics == (_legacy_derived_name(spec.url_pattern) in INV_EXTR_DATASETS)
    for spec in TEST_SPLITS.values()
), "TEST_SPLITS.invert_extrinsics disagrees with views.INV_EXTR_DATASETS"


def seed_everything(seed: int) -> None:
    """Reseed every RNG this pipeline's reproducibility depends on.

    `random`/numpy are the obvious ones (the view shuffle and the
    WebDataset shard-shuffle seed draw both read Python's global `random`).
    Torch is seeded too even though no tensor op here is random: with
    `workers > 0`, `torch.utils.data.DataLoader` draws this worker
    generation's `base_seed` from torch's *current* global RNG state at
    `iter(loader)` time, and `_worker_init_fn` derives each worker's
    `random`/numpy seed from that (`torch.initial_seed()`). Leaving torch
    unseeded here does not break any single run, but it breaks run-to-run
    and legacy-vs-`uafit` reproducibility for `workers > 0`: torch's RNG
    keeps drifting across calls, so each `build_test_loader(...)` call's
    workers land on different per-sample view shuffles / master-view
    choices than before -- verified by this exact symptom in the
    `workers=4` parity check before this line was added."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Ported from `lib.utils.net_utils.worker_init_fn`: PyTorch's
    `DataLoader` already assigned each forked worker its own torch seed
    before calling this; derive this worker's `random`/numpy seed from it so
    workers don't all draw the identical shuffle."""
    seed = (worker_id + int(torch.initial_seed())) % _INT32_MOD
    np.random.seed(seed)
    random.seed(seed)


class _TestIterable(IterableDataset):
    """Reads `split`'s shards and yields `views.process_sample` dicts,
    stopping after `limit` scenes (0 = the full split).

    `open_shards` is called once here, in `__init__` -- i.e. in the main
    process, before `DataLoader` forks any workers -- not lazily inside
    `__iter__`. This matters for `workers > 0`: `open_shards` draws one
    `random.randint` to seed that dataset's shard-order shuffle (see
    `shards.open_shards`'s docstring). The legacy pipeline draws this seed
    the same way, eagerly building its dataset object (`create_dataset(...)`
    in `scripts/eval.py`) before handing it to `wds.WebLoader`; every forked
    worker then inherits the *same* seed via `fork()`. Drawing it lazily
    inside `__iter__` instead would draw it separately per worker, *after*
    `_worker_init_fn` has already given each worker its own distinct
    `random` stream -- silently reordering which shards each worker reads
    first and breaking run-to-run and legacy-vs-`uafit` reproducibility for
    any multi-shard-per-worker split (verified: this was a real bug caught
    by the `workers=4` parity check, not a hypothetical one).
    """

    def __init__(self, split: TestSplit, data_root: str, n_views: int, limit: int):
        self.split = split
        self.n_views = n_views
        self.limit = limit
        self._stream = open_shards(split.url_pattern, data_root)

    def __iter__(self):
        n = 0
        for item in self._stream:
            raw = group_views(item)
            yield process_sample(
                raw,
                self.n_views,
                self.split.crop_size,
                self.split.recrop_from_joints,
                self.split.invert_extrinsics,
            )
            n += 1
            if self.limit and n >= self.limit:
                return


def build_test_loader(
    dataset: str,
    data_root: str,
    max_views: Optional[int] = None,
    limit: int = 0,
    batch_size: int = 16,
    workers: int = 4,
    seed: int = 1,
) -> DataLoader:
    """Rebuild one of Table 1's test/val loaders from the released shards.

    Args:
        dataset: one of `TEST_SPLITS` (`"DexYCB"`, `"HO3D"`, `"ARCTIC"`,
            `"InterHand"`, `"OakInk"`).
        data_root: the `dataset_tars` directory itself (parent of each
            `TestSplit.url_pattern`, e.g. `.../data/dataset_tars`).
        max_views: cap on views/scene; `None` uses the split's fixed count
            (DexYCB/ARCTIC/InterHand 8, HO3D 5, OakInk 4).
        limit: stop after this many scenes *per worker* (so the true cap is
            up to `limit * max(workers, 1)`); `0` evaluates the full split.
            A sanity-check knob (mirrors `scripts/eval.py --limit`, which
            caps scenes *scored*, not read) -- for an exact global cap, use
            `workers=0`.
        seed: reseeds this process (`seed_everything`) immediately before
            building the shard stream.

    Determinism: call this right after `seed`-ing, as `scripts/eval.py`
    does via `setup_seed`. The per-sample view order (`views.select_views`)
    and the shard read order both depend on the ambient `random` state *at
    construction time*, not on `seed` alone -- this function reseeds for
    you so that ambient state is exactly `seed`.
    """
    if dataset not in TEST_SPLITS:
        raise KeyError(f"unknown test split {dataset!r}; choose from {sorted(TEST_SPLITS)}")
    split = TEST_SPLITS[dataset]
    n_views = min(max_views, split.n_views) if max_views else split.n_views

    seed_everything(seed)
    iterable = _TestIterable(split, data_root, n_views, limit)
    return DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        collate_fn=collate_views,
    )
