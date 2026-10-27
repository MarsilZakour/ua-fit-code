"""Per-scene view selection and per-view crop/normalise.

This is the test-time (`AUG: false`) slice of
`lib.data_wds.multiview_wds.MultiviewWebDataset.process_data_item` and
`lib.utils.transform.SimpleTransform3DMultiView`. At `AUG: false` the
in-plane rotation jitter is always 0, which collapses
`_affine_transform`/`_affine_transform_post_rot` (two different-looking
formulas in the legacy code) onto the exact same no-rotation crop affine --
ported once here as `crop_affine`. No geometric/photometric/colour jitter,
occlusion, or horizontal flip is ported: every one of those branches is dead
code when `AUG: false`, which every released test/val split config sets.
"""
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tvf

# Dataset names (the shard file's basename up to the first '_') whose stored
# `cam_extr` must be inverted before use. Ported verbatim from
# `lib.data_wds.multiview_wds.INV_EXTR_DATASETS` -- these were silently
# dropped once by an uncommitted edit and un-inverting them corrupted all
# Interhand/OakInk multi-view geometry (POEM-large InterHand 8-view: 7.1 mm
# -> 212 mm MPJPE), so treat this set as load-bearing, not cosmetic.
INV_EXTR_DATASETS = frozenset({"Interhand", "Arctic", "Oakink", "Oakink2"})


def select_views(n_cams: int, n_views: int) -> List[int]:
    """Pick which of the `n_cams` cameras go into this scene's batch, and
    which becomes the "master" view (index 0 of the result -- its frame is
    the scene's 3D reference frame, i.e. `master_joints_3d`/`master_verts_3d`
    and every `target_cam_extr` are expressed relative to it).

    Surprising, verified-against-the-checkpoint behaviour: every released
    eval config section keeps `RANDOM_N_VIEWS: true` with `VIEW_RANGE =
    [n_views, n_views]`, so the view *count* is pinned but the master view
    (and the order of the kept cameras) is still drawn uniformly at random
    per scene -- it is never camera index 0. Exactly replaying a legacy run
    (for the parity test, or to debug a specific scene) requires consuming
    Python's global `random` state in the identical order:
      1. one `random.random()` -- legacy wraps even a single test split in
         `wds.RandomMix([dataset], [1.0])`; the draw always selects the only
         source but still advances the stream. Kept here as a deliberate
         no-op so the two streams stay in lockstep.
      2. `random.shuffle(indices)` -- the actual view permutation.
      3. `random.gauss(4, 2)` -- the legacy "how many views" draw. Its
         result is discarded: with `VIEW_RANGE = [n_views, n_views]` the
         count is clamped to `n_views` regardless of what was drawn. Still
         must be called, or every later sample's `random` state desyncs.
    """
    indices = list(range(n_cams))
    random.random()
    random.shuffle(indices)
    random.gauss(4, 2)
    return indices[: min(n_views, n_cams)]


def recrop_bbox_from_joints(joints_2d: np.ndarray, joints_vis: np.ndarray) -> Tuple[np.ndarray, np.float32]:
    """Rebuild the crop box from visible 2D joints instead of the stored
    `bbox_center`/`bbox_scale`. ~21% of InterHand2.6M's released val
    `bbox_scale` values are 10-50x too large (the hand shrinks to a dot in
    the crop); every method in Table 1 is scored with this fix applied, via
    `TestSplit.recrop_from_joints`."""
    vis = joints_vis.reshape(-1) > 0.5
    sel = joints_2d[vis] if int(vis.sum()) >= 3 else joints_2d
    lo, hi = sel.min(0), sel.max(0)
    center = ((lo + hi) / 2.0).astype(np.float32)
    scale = np.float32(max(float((hi - lo).max()) * 2.0, 60.0))
    return center, scale


def crop_affine(center: np.ndarray, scale, out_size: int) -> np.ndarray:
    """3x3 affine mapping the `scale` x `scale` pixel box centred at
    `center` onto an `out_size` x `out_size` output, with no rotation.

    Ported from `lib.utils.transform._get_affine_trans_no_rot` (its square
    `out_size` specialisation: the general code's `scale_ratio =
    res[0]/res[1]` term is 1 since every release config uses a square
    `DATA_PRESET.IMAGE_SIZE`). This is also, bit-for-bit, what
    `_affine_transform` (image crop) and `_affine_transform_post_rot`
    (target intrinsics) both reduce to at `rot=0`: both route `center`
    through one or two rotate-about-a-point compositions that are exact
    identities at `rot=0` (rotating by the identity matrix, or translating
    by `-c` then back by `+c`), so using one shared affine for both the
    image warp and the intrinsics update reproduces the legacy two-call path
    exactly, not just equivalently.
    """
    affine = np.zeros((3, 3), dtype=np.float32)
    affine[0, 0] = float(out_size) / scale
    affine[1, 1] = float(out_size) / scale
    affine[0, 2] = out_size * (-float(center[0]) / scale + 0.5)
    affine[1, 2] = out_size * (-float(center[1]) / scale + 0.5)
    affine[2, 2] = 1.0
    return affine


def build_view(
    image: np.ndarray,
    cam_intr: np.ndarray,
    cam_extr: np.ndarray,
    cam_extr_master: np.ndarray,
    bbox_center: np.ndarray,
    bbox_scale,
    out_size: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """One selected camera -> `(image, target_cam_intr, target_cam_extr)`.

    `image`: normalised CHW float tensor, bit-for-bit
    `SimpleTransform2D.__call__`'s `AUG: false` branch (`to_tensor(img) -
    0.5`, no colour/photometric jitter). `target_cam_extr` expresses this
    view's extrinsic relative to the scene's master view -- ported from
    `MultiviewWebDataset.process_data_item`'s extrinsic composition,
    including its algebraically-redundant double matrix inversion (see the
    comment at the `target_cam_extr` line below: it is not a bit-exact
    no-op, so it is kept).
    """
    affine = crop_affine(bbox_center, bbox_scale, out_size)
    warped = cv2.warpAffine(
        image, affine[:2], (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    image_t = tvf.normalize(tvf.to_tensor(warped), [0.5, 0.5, 0.5], [1, 1, 1])
    target_cam_intr = (affine @ cam_intr).astype(np.float32)
    # Mathematically `inv(identity @ inv(view_rel_extr)) == view_rel_extr`, so
    # the identity-rotation compose the legacy code performs here (its
    # `extr_pre_transf`, always identity at rot=0) looks like a no-op and
    # could be dropped. It is NOT a bit-exact no-op: `np.linalg.inv` run
    # twice on a float32 matrix does not generally round-trip to the exact
    # same bits (observed ~1-2 float32 ULP, e.g. 6e-8 on a ~1.0-scale
    # entry). Replicated verbatim so this function reproduces the released
    # eval numerically, not just algebraically.
    view_rel_extr = np.linalg.inv(cam_extr_master) @ cam_extr
    target_cam_extr = np.linalg.inv(np.linalg.inv(view_rel_extr)).astype(np.float32)
    return image_t, target_cam_intr, target_cam_extr


def process_sample(
    raw: Dict,
    n_views: int,
    crop_size: int,
    recrop_from_joints: bool,
    invert_extrinsics: bool,
) -> Dict:
    """`shards.group_views` output -> one collate-ready sample.

    Applies `select_views`, the optional InterHand re-crop, and
    `build_view` per kept camera, then attaches the scene-level GT
    (`master_joints_3d`/`master_verts_3d`, from the master view's *own*
    un-cropped `joints_3d`/`verts_3d` -- these already live in the master
    camera's frame in the raw annotation, so no further transform is
    needed) and the per-view MANO shape GT.
    """
    labels = raw["labels"]
    keep = select_views(raw["n_cams"], n_views)
    master = keep[0]

    cam_extr = labels["cam_extr"]
    if invert_extrinsics:
        cam_extr = [np.linalg.inv(e) for e in cam_extr]
    cam_extr_master = cam_extr[master]

    images, intrs, extrs = [], [], []
    for v in keep:
        center, scale = labels["bbox_center"][v], labels["bbox_scale"][v]
        if recrop_from_joints:
            center, scale = recrop_bbox_from_joints(
                np.asarray(labels["joints_2d"][v], dtype=np.float32),
                np.asarray(labels["joints_vis"][v], dtype=np.float32),
            )
        image_t, intr, extr = build_view(
            raw["images"][v], labels["cam_intr"][v], cam_extr[v], cam_extr_master, center, scale, crop_size
        )
        images.append(image_t)
        intrs.append(intr)
        extrs.append(extr)

    return {
        "image": torch.stack(images),  # (n_views, 3, crop_size, crop_size)
        "target_cam_intr": np.stack(intrs),  # (n_views, 3, 3)
        "target_cam_extr": np.stack(extrs),  # (n_views, 4, 4)
        "sample_idx": np.asarray([labels["sample_idx"][v] for v in keep], dtype=np.int64),  # (n_views,)
        "target_mano_shape": np.stack(
            [np.asarray(labels["mano_shape"][v], dtype=np.float32) for v in keep]
        ),  # (n_views, 10)
        "master_joints_3d": np.asarray(labels["joints_3d"][master], dtype=np.float32),  # (21, 3)
        "master_verts_3d": np.asarray(labels["verts_3d"][master], dtype=np.float32),  # (778, 3)
    }
