"""Read the released multi-view WebDataset `.tar` shards.

Each shard sample groups `n_cams` views as `image_<i>.<jpg|png>` members plus
one `label.pyd` (a pickled dict of length-`n_cams` lists: `cam_intr`,
`cam_extr`, `joints_2d`, `joints_3d`, `verts_3d`, `joints_vis`, `bbox_center`,
`bbox_scale`, `sample_idx`, `mano_shape`, ...). This module only opens the
shards and regroups that layout into plain per-view lists; no view
selection, cropping or GT-frame composition happens here (see `views.py`).
"""
from typing import Dict, List

import numpy as np
import torch
import webdataset as wds
from braceexpand import braceexpand
from torchvision.io import ImageReadMode, decode_jpeg


def _decode_jpeg(key: str, data: bytes):
    """Fast libjpeg-turbo decode -> contiguous HWC uint8 RGB numpy array.
    Non-JPEG members (and the `label.pyd` pickle) return `None`, falling
    through to WebDataset's built-in extension-based decoders."""
    if key.endswith(".jpg") or key.endswith(".jpeg"):
        raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        img = decode_jpeg(raw, mode=ImageReadMode.RGB)  # (3, H, W) uint8
        return np.ascontiguousarray(img.permute(1, 2, 0).numpy())
    return None


def expand_urls(url_pattern: str, data_root: str) -> List[str]:
    """Brace-expand a shard glob (e.g. ``"DexYCB_mv/DexYCB_mv_test-{0..3}.tar"``)
    rooted at `data_root`, e.g. ``.../data/dataset_tars``."""
    full_pattern = f"{data_root.rstrip('/')}/{url_pattern}"
    return list(braceexpand(full_pattern))


def open_shards(url_pattern: str, data_root: str) -> wds.WebDataset:
    """Open `url_pattern`'s shards as a decoded WebDataset sample stream.

    Uses the same `shardshuffle=True` (no `detshuffle`) construction the
    released pipeline uses for every split, train or test: it draws one
    `random.randint` from the ambient global `random` state to seed a
    shard-order shuffle. That draw -- and every later per-sample draw in
    `views.select_views` -- must come from the same `random` stream as the
    pipeline being compared against for the two pipelines' sample order to
    match; see `loader.build_test_loader`.
    """
    urls = expand_urls(url_pattern, data_root)
    dataset = wds.WebDataset(
        urls=urls,
        nodesplitter=wds.split_by_node,
        shardshuffle=True,
        resampled=False,
        empty_check=False,
    )
    return dataset.decode(_decode_jpeg, "rgb8")


def group_views(item: Dict) -> Dict:
    """One decoded tar sample -> `{"key", "n_cams", "images", "labels"}`.

    `images` is a list of `n_cams` HWC uint8 arrays in camera-index order;
    `labels` is the sample's `label.pyd` dict (per-view python lists).
    OakInk shards carry no MANO fit (`mano_shape` absent); it is filled with
    zeros so every dataset's sample dict has the same keys.
    """
    image_keys = sorted(
        (k for k in item if k.startswith("image_")),
        key=lambda k: int(k.split(".")[0].split("_")[1]),
    )
    labels = dict(item["label.pyd"])
    n_cams = len(image_keys)
    if "mano_shape" not in labels:
        labels["mano_shape"] = [np.zeros(10, dtype=np.float32) for _ in range(n_cams)]
    return {
        "key": item["__key__"],
        "n_cams": n_cams,
        "images": [item[k] for k in image_keys],
        "labels": labels,
    }
