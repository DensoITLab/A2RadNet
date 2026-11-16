# Reproduction scope and implementation decisions

This candidate preserves the proposed-method math and recorded seed-3 configs.
Public imports and machine paths have been cleaned. No parent checkout is required.
The original source hashes are in `source_manifest.json` for the maintainer's audit.
The real-data benchmarks are included; synthetic analysis and ablation variants are
outside this initial proposed-method release.

## Recorded configurations

| Item | RADIal | RADDet |
|---|---|---|
| Input `[2C,R,D]` | `[32,512,256]` | `[16,256,64]` |
| Pseudo-angle bins Ap | **112** | 128 |
| Remaining channels L | 32 DLR tokens | 64 Doppler bins |
| AAE projection | Circular, dilation 16, kernel width 16 | Pointwise |
| Spatial attention window | `[14,4]` in all stages | `[14,4]` in all stages |
| Detection `[channels,R,A]` | `[3,128,112]` | `[3,64,128]` |
| Additional output | Freespace `[1,256,224]` | Class logits `[6,64,128]` |
| Training batch size | 4 | 16 |
| Seed | 3 | 3 |
| Historical evaluation epoch | 118 (zero based) | 141 (zero based) |

The first experiment listed as `Ours` in each existing seed-summary script was used.
Those scripts label seed positions 0/1/2; their actual saved config seed values must
be used instead. This initial validation covers the seed-3 reference runs.

## Confirmed implementation decisions

The maintainer has confirmed that the implementation is authoritative. The
following choices are resolved; model shapes and computations remain unchanged.
The original paper PDF is not modified by this release preparation.

1. **RADIal pseudo-angle count:** `Ap=112`. The paper's `Ap=128` sentence does not
   describe the selected trained implementation and should be corrected in paper
   text when appropriate. The RADDet implementation uses `Ap=128`.
2. **RADIal split:** preserve the recorded fixed sequences, yielding
   **6231/986/1035** samples (approximately 75.5/11.9/12.5 percent). These actual
   counts take precedence over the paper table's 70/15/15 description. The ratio
   field in the historical config is ignored in sequence mode.
3. **DLR:** learned queries, Q/K/V projections, attention and an output linear
   projection, with **no separate post-attention FFN**. Do not add an FFN to the
   trained architecture to match the paper wording.
4. **Normalization:** the maintainer confirms that the stored normalization
   statistics were computed using training data only. Validation and test data
   were excluded. Recorded statistics are retained.

## Checkpoint selection: resolved

The historical evaluation epochs are **118 for RADIal and 141 for RADDet**.
Both agree with maximum validation F1 computed from the complete 150-epoch
histories (RADIal excludes epochs 0–19). The original selector currently reads
epoch 149 by filesystem change time; the public selector reads it by numeric
epoch. Both choose the same evaluation checkpoints on the inspected runs.

An earlier release-preparation audit incorrectly loaded epoch 99 after sorting
checkpoint filenames lexicographically: `epoch99` sorts after `epoch149`.
Its truncated 100-epoch histories yielded 98/70. Those were an audit error, not
a disagreement in the saved experiments or the public selector. The corrected
audit is recorded in `checkpoint_selection_audit.json`.

## Remaining numerical validation

Full pretrained test evaluations complete for both datasets, but small differences
from historical metric JSON files remain; see [validation.md](validation.md).
A complete fresh seed-3 150-epoch training followed by test evaluation is still
outstanding. Dataset archive versions and clean installation also remain to be
verified. These are separate from the resolved architecture, normalization and
checkpoint-selection decisions.

## Data and split protocol

RADIal uses precomputed complex RD arrays. Channels are concatenated as all real
components followed by all imaginary components, then normalized with config values.
All labelled frames, including difficult frames, are loaded. Validation and test
sequences are fixed in `a2radnet/radial/dataloader.py`; all other sample IDs are used
for training. No point clouds or auxiliary LiDAR losses are used.

RADDet loads complex ADC arrays, removes the mean over range/chirps, applies the
existing Hamming coefficients before range/Doppler FFT, and fftshifts the Doppler
axis. The historical variable names say `hanningWindow`, but the formula is Hamming.
The official training partition is split 90/10 using a seeded PyTorch generator;
the official test annotations remain separate. Installed counts are **7313/813/2032**.
Class names use the alphabetically sorted scikit-learn LabelEncoder order.

Splits are checked for overlapping sample IDs and written to runtime JSON files.
The installed annotation fingerprints and exact seed-3 split IDs are recorded in
`reference_results/`; `scripts/check_data.py` verifies them. Full radar file contents
and original download archive hashes are not pinned. Dataset preparation must retain
original sample IDs; synthetic random test partitions are not substitutes.

## Training and metric compatibility

The original loss scaling, optimizer, scheduler, and validation schedules are
preserved. In particular, the config value `smoothL1Loss` does not match the
inherited loss code's case-sensitive `SmoothL1Loss` branch: the recorded run uses
**L1** regression. Renaming that config value would change training behavior.
Freespace BCE retains the original double-precision target and batch scaling.

Binary detection `Precision`/`Recall` are the inherited benchmark aggregates over
confidence thresholds. The original table scripts call them AP/AR; these are not
silently replaced with COCO AP or another standard interpolated AP implementation.
RADDet class-wise AP/mAP also retains the inherited benchmark implementation.
NMS, box conversion, thresholds, errors and angle conventions are unchanged.
RADIal evaluates detection in the 5–100 m range and freespace over the first 124
rows of the output grid; RADDet evaluates detection in 0–50 m.

Seeds initialize Python, NumPy and PyTorch, but deterministic GPU algorithms are
not forced. Environment and cuDNN settings are recorded. Exact bitwise agreement
across hardware, library versions, or resumed training is not guaranteed.

## Source changes

- Retain only the proposed A2RadNet model, its AAE/DLR, spatial Swin, loaders,
  encoders, losses, and benchmark metrics.
- Rename public classes/modules to AAE, DLR and RPA terminology; retain learned
  parameter attribute paths to load historical checkpoints strictly.
- Remove comparison-model imports, unused periodic-attention branches, complex
  ADC model dependencies, workstation fallback paths, and point-cloud model code.
- Preserve Microsoft Swin attribution and include its upstream MIT license.
- Use explicit CLI inputs, record execution metadata/splits, and select checkpoint
  epochs numerically instead of relying on filesystem creation times or substrings.
- Use zero data-loader workers by default after a worker segmentation fault in
  MKL-backed RADDet FFT preprocessing; permit explicit worker overrides.

Local exports deliberately omit optimizer state, config paths, TensorBoard events,
hostnames, usernames, private scripts and dataset files. Local runtime logs contain
execution paths and are excluded by `.gitignore`; do not include `outputs/` in a release.

## Removal of unused auxiliary paths

The release excludes the experimental auxiliary map head and loss, its projection
output, and the associated config fields. RADIal loss weights are now the three
active weights `[2, 100, 100]`; the inactive fourth zero and `occupancy` option
have been removed. Legacy configs with trailing zero weights still load.

Dataset/collate results no longer contain empty point-cloud/map placeholders:
RADIal samples and batches contain the five existing radar/target/box/image fields;
RADDet samples and batches contain the four existing radar/target/box/class fields.
The original task-field order, preprocessing, task heads, normalization, splits
and benchmark metric definitions are preserved. Model output no longer includes
the unused `raw_ra` auxiliary entry. Feature inspection via `return_features=True`
remains available. Existing pretrained state dictionaries still load strictly.
