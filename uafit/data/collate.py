"""Batch collation: concatenate each scene's per-view fields along the view
axis and record how many views each scene contributed. Narrow counterpart of
`lib.utils.collation.collation_random_n_views`, restricted to the keys this
pipeline emits (see `views.process_sample`).
"""
from typing import Dict, List

import numpy as np
import torch


def collate_views(samples: List[Dict]) -> Dict:
    """`samples`: one `views.process_sample` dict per scene. Returns a
    batch dict with every per-view field concatenated along axis 0 (total
    views in the batch), `master_*` stacked along a fresh batch axis `(B,
    ...)`, and `cam_view_num`: `(B,)` int64 views-per-scene -- kept as plain
    numpy (not a tensor), matching how `uafit.network`/`eval.py` index with
    it (`np.cumsum(batch["cam_view_num"])`, scene boundaries into the
    view-concatenated tensors).
    """
    return {
        "image": torch.cat([s["image"] for s in samples], dim=0),
        "target_cam_intr": torch.as_tensor(
            np.concatenate([s["target_cam_intr"] for s in samples], axis=0)
        ),
        "target_cam_extr": torch.as_tensor(
            np.concatenate([s["target_cam_extr"] for s in samples], axis=0)
        ),
        "sample_idx": torch.as_tensor(
            np.concatenate([s["sample_idx"] for s in samples], axis=0)
        ),
        "target_mano_shape": torch.as_tensor(
            np.concatenate([s["target_mano_shape"] for s in samples], axis=0)
        ),
        "master_joints_3d": torch.as_tensor(np.stack([s["master_joints_3d"] for s in samples], axis=0)),
        "master_verts_3d": torch.as_tensor(np.stack([s["master_verts_3d"] for s in samples], axis=0)),
        "cam_view_num": np.asarray([s["image"].shape[0] for s in samples], dtype=np.int64),
    }


def batch_to_device(batch: Dict, device) -> Dict:
    """Move a collated batch to ``device`` (floating tensors as float32); ``cam_view_num`` stays a numpy array."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device).float() if value.is_floating_point() else value.to(device)
        elif isinstance(value, np.ndarray) and key != "cam_view_num":
            out[key] = torch.as_tensor(value).to(device)
        else:
            out[key] = value
    return out
