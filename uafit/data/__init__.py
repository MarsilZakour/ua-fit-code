"""Test-time multi-view data pipeline for UA-Fit's Table 1 evaluation.

(References to ``lib.*`` name the original research implementation this was ported
from; it is kept in the git history, see REPRODUCE.md.)

Reads the released WebDataset `.tar` shards directly and reproduces the
*test* path of the legacy pipeline (`lib.data_wds.multiview_wds` +
`lib.utils.transform.SimpleTransform3DMultiView`, driven from
`scripts/eval.py`). Training-only concerns (geometric/photometric
augmentation, dataset mixing, the raw non-shard dataset classes in
`lib/datasets/`) are intentionally not ported -- they never run when
`AUG: false`, which every released test/val config section sets.

Public API:
    `build_test_loader(dataset, data_root, ...)` -- a `torch.utils.data.DataLoader`
        yielding batches with exactly the keys `uafit.network` /
        `uafit.fitting` consume (see `collate.py`).
    `TEST_SPLITS` -- registry of the 5 Table-1 splits (shard glob, fixed view
        count, official split size, ...).
"""
from .collate import batch_to_device
from .loader import TEST_SPLITS, TestSplit, build_test_loader

__all__ = ["TEST_SPLITS", "TestSplit", "batch_to_device", "build_test_loader"]
