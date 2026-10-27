# Assets

| path | needed for | how to get it |
|---|---|---|
| `mano_v1_2/models/MANO_RIGHT.pkl` | everything | register and download MANO v1.2 from https://mano.is.tue.mpg.de/ (license-restricted, not redistributed) |
| `correspondences/mhr_right.npz`, `correspondences/umetrack_right.npz` | `--hand-model mhr / umetrack` | shipped; rebuild with `scripts/build_mano_correspondence.py` |
| `mhr/mhr_lod1_cache.pt` | `--hand-model mhr` | a torch cache of Meta's MHR (Momentum Human Rig, LOD 1), written by parafit's `MhrModel.save_cache` (see below) |
| `umetrack/generic_hand_model.json` | `--hand-model umetrack` | `dataset/generic_hand_model.json` of https://github.com/facebookresearch/UmeTrack (CC BY-NC 4.0, not redistributed) |
| `media/` | README | |

## MHR cache

parafit re-implements the MHR forward in plain torch and loads it from a single
cache file, so evaluation needs neither `pymomentum` nor the MHR asset folder.
Build the cache once on a machine with MHR installed
(https://github.com/facebookresearch/MHR, `pip install pymomentum-gpu mhr`):

```python
import pymomentum.geometry  # must be imported before torch
from parafit.models.mhr import MhrModel
MhrModel(asset_folder="<MHR assets>", lod=1, device="cpu").save_cache("assets/mhr/mhr_lod1_cache.pt")
```

## MANO correspondences

`correspondences/<model>_right.npz` attach each of MANO's 21 joints (16
mesh-regressed joints + 5 fingertip vertices, OpenPose order) to one bone of the
target hand with a fixed offset, and map every MANO vertex to a barycentric point
on the target mesh. They are calibrated on synthetic MANO hands (PCA pose prior,
random shapes) by alternating fits and closed-form re-attachment; the file's
`calibration` entry records the held-out residuals. Rebuild:

```bash
python scripts/build_mano_correspondence.py --model mhr
python scripts/build_mano_correspondence.py --model umetrack
```
