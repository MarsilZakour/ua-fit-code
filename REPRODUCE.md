# Reproducing UA-Fit Table 1

## Protocol

Every number is measured through one harness at seed 1:

- **Full official split** of each dataset at a fixed view count (DexYCB 8,
  HO3D 5, ARCTIC 8, InterHand2.6M 8, OakInk 4). Which cameras form the views of a
  scene is drawn from the seeded RNG, exactly as in the original evaluation.
- **Feed-forward shape ("fast") mode**: the shape head predicts the MANO betas and
  the analytical Levenberg-Marquardt solver performs a single fit (15 iterations,
  3D-anchor weight `lambda_3D = 3e5`, learned Huber radius scaled by 0.5, i.e.
  3 px -> 1.5 px). All settings: `uafit.fitting.FitSettings`.
- **Metrics** (mm): MPVPE, MPJPE against **mesh-regressed** ground-truth joints
  (the same regressor defines predicted and ground-truth joints, because the raw
  joint annotations follow other conventions), PA-MPVPE.
- **InterHand crop fix**: part of the released InterHand2.6M boxes are corrupt
  (out-of-frame joints inflate them); every view's crop is rebuilt from the
  visible ground-truth joints, identically for every method.

## Data

The test shards are the multi-view WebDataset release of POEM-v2
(https://huggingface.co/datasets/JubSteven/POEM-v2); see the README for the download command (test shards only).

## Full table

```bash
python scripts/evaluate.py --output runs/table1_mano.json
```

| dataset (views) | n     | MPVPE | MPJPE | PA-MPVPE |
|---|---|---|---|---|
| DexYCB (8v)    | 4931  | 7.86 | 7.55 | 4.92 |
| HO3D (5v)      | 2687  | 8.01 | 7.41 | 5.10 |
| ARCTIC (8v)    | 17359 | 7.54 | 6.62 | 5.66 |
| InterHand (8v) | 85235 | 8.77 | 7.79 | 6.23 |
| OakInk (4v)    | 21322 | 9.58 | 8.86 | 5.67 |

Runtime: about 1.5 h for all five splits on one (shared) H200; InterHand2.6M dominates.

## Per-dataset and quick runs

```bash
python scripts/evaluate.py --datasets HO3D                    # one dataset, full split
python scripts/evaluate.py --datasets HO3D DexYCB --limit 200  # quick subset (ARCTIC / InterHand are subset-sensitive)
python scripts/evaluate.py --views 4                           # cap every dataset at 4 views
python scripts/evaluate.py --iterations 5                      # solver settings: --iterations, --joint-anchor-weight, --huber-radius-scale
```

Quick-subset reference (`--datasets HO3D DexYCB --limit 200`, 208 scenes each):
HO3D 7.61 / 7.16 / 4.66, DexYCB 7.38 / 7.07 / 4.73.

## Other hand models (examples)

`--hand-model mhr` / `--hand-model umetrack` fit Meta's MHR hand or the UmeTrack hand to the same network output
through a calibrated MANO-joint correspondence. They are examples of the model-agnostic solver, not part of the
paper; see the README.

## How the release code was checked against the original

The release package (`uafit/`) is a re-implementation of the research code that
produced the paper numbers (`lib/`, `scripts/eval.py`; still in the git history
at commit `50d9efb`, together with the parity scripts in `tests/`; run them from a checkout of that commit):

- `tests/test_network_parity.py`: every network output (triangulated joints,
  betas, vote field, vote precision, heatmap intrinsics, extrinsics) is
  bit-identical to the original network on HO3D and DexYCB batches.
- `tests/test_data_parity.py`: test batches are bit-identical to the original
  loader for all five datasets (single worker), and the scene sets match with 4
  workers.
- `tests/test_fitter_parity.py`: the solver (parafit `LMSolver` + `uafit.fitting`
  energies + parafit `ManoModel`) builds the same normal equations as the original
  solver (relative difference ~1e-7). The paper solves them in float32 at
  condition numbers ~1e9, so rounding at that level moves individual samples; the
  test measures that noise floor by re-running the original solver on inputs
  perturbed by 1e-6 px. On 200 scenes each of HO3D, DexYCB and ARCTIC the release
  solver's per-sample deviation equals that floor (mean 0.01-0.02 mm) and the mean
  MPVPE agrees within 0.002 mm.
