# Validation status

Validation used the inspected Python 3.9 / PyTorch 1.12.1+cu113 environment.
These are completed checks, not a claim of full paper reproduction.

| Check | RADIal | RADDet |
|---|---|---|
| Recorded seed | 3 | 3 |
| Strict model/checkpoint loading | Pass | Pass |
| Full test evaluation | 1035 frames | 2032 frames |
| Explicit checkpoint epoch (zero based) | 118 | 141 |
| Same-input outputs vs current original model | Exact equality, all output tensors | Exact equality, all output tensors |
| Real-data forward/backward + validation + save | Pass (1 batch each) | Pass (1 batch each) |
| Resume optimizer/scheduler/history | Pass (epoch 0 → 1, step 1 → 2) | Pass (epoch 0 → 1, step 1 → 2) |
| Fresh 150-epoch single-seed training | **Not run** | **Not run** |
| All historical JSON fields within 1e-5 | **Fail** | **Fail** |

## Fresh full-test results

| Metric | RADIal | RADDet |
|---|---|---|
| Binary precision at IoU 0.5 | 97.049266% | 73.787641% |
| Binary recall at IoU 0.5 | 94.673721% | 68.097339% |
| Range error | 0.111985665 m | 0.215511211 m |
| Angle error | 0.082663826 deg | 0.195110588 deg |
| Freespace mIoU | 83.272338% | — |
| Class-wise mAP | — | 71.457371% |

Historical JSON files are retained as `reference_results/*_seed3.json`.
Fresh results are separately stored as `reference_results/*_verified_seed3.json`.
Do not replace the historical reference with newly measured values to make a
reproduction check pass. `validation_report.json` records per-field discrepancies
and checkpoint hashes without machine paths.

At IoU 0.5, RADIal precision/recall equal the historical values; errors and mIoU
differ by less than 1e-6. Across all IoU thresholds, RADIal has a difference of
about 0.081 percentage points in precision at IoU 0.8. RADDet differs by about
0.040 percentage points in precision at IoU 0.5. The cause of the historical
metric differences is **not established**. Current source-model equivalence is
verified; this alone does not establish historical environment/data equivalence.
The current dataset annotation fingerprints and exact splits are recorded.

The strict comparison deliberately reports failure against the historical files.
No tolerance was loosened to hide these differences. Re-evaluating the exported
checkpoint on this environment can be compared with the separate verified file,
while historical reconstruction remains an explicit follow-up.

## Other completed checks

- Six contract tests pass: periodic padding, DLR permutation invariance/gradients,
  both AAE variants' RPA layouts/gradients, validation checkpoint selection, and
  the simplified RADIal/RADDet task batch contracts.
- A standalone copy in a temporary directory passes command help, imports, unit
  tests and the source audit without the parent repository on the Python path.
- Real-data annotation fingerprints, split membership, disjointness, and required
  file presence are checked with `scripts/check_data.py` for both datasets.
- Sanitized checkpoint exports contain only CPU model tensors, epoch, and seed;
  source/public hashes are recorded next to each export. They strictly load into
  both the current original models and the standalone models.
- The inspected original 114 Python/config/documentation/shell files retain their
  hashes. All deliverables and runtime artifacts are added under this candidate.

The first RADDet evaluation attempt crashed in a forked MKL FFT worker. The full
rerun with zero workers completed. The RADIal full evaluation used one worker;
subsequent smoke checks and the candidate defaults use zero. Metadata records the
actual worker settings.

## Remaining before a final public release

Architecture/configuration choices and training-only normalization are confirmed
by the maintainer; the implementation is authoritative. Checkpoint selection is
resolved: the complete histories select epochs 118/141. See
[reproduction.md](reproduction.md) and `checkpoint_selection_audit.json`.

Remaining work is to investigate historical metric differences, verify a clean
environment installation, and perform the agreed full single-seed training and
test evaluation. The ITLAB non-commercial license has been selected for original code contributions
and the provided pretrained models; inherited-code terms are described in
[licensing.md](licensing.md). Dataset
archive versions, full training time and minimum VRAM remain unmeasured.

## Auxiliary-path cleanup verification

The unused auxiliary map model/loss and empty point-cloud batch fields were
removed. Existing model weights load strictly, and both datasets produce bitwise
identical task outputs on the same real test sample before and after cleanup.
The task batch tensors and training loss also match exactly; gradients remain
finite. GPU backward determinism is not enforced. Details are recorded in
[auxiliary_cleanup_report.json](auxiliary_cleanup_report.json).
