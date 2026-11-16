# Detailed usage

Run commands from the repository root. For the short introduction, see
[README](../README.md). This page includes Docker execution, advanced commands,
and implementation details.

## Environment

The inspected machine has Ubuntu 20.04.6, Python 3.9.5, PyTorch 1.12.1+cu113,
torchvision 0.13.1+cu113, CUDA Toolkit 11.3, and RTX 6000 Ada 48 GB GPUs.
Each command uses one GPU. This GPU capacity is the inspected hardware, not a
measured minimum requirement. Use Python 3.9 for the supplied dependency pins.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

These commands run from this repository root. The current environment has been
inspected and used for validation. The maintainer also built the supplied Dockerfile
in a separate environment and confirmed GPU execution; the RADIal pretrained
evaluation matched the historical seed-3 results within absolute tolerance 1e-5. CUDA in the PyTorch wheel, the installed Toolkit, and
the GPU driver are separate versioned components. See [environment details](environment.md).

## Docker

Build from the repository root. GPU execution requires Docker and NVIDIA Container
Toolkit configured on the host.

```bash
docker build -t a2radnet:verify .
docker run --rm --gpus device=0 a2radnet:verify \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The image uses Python 3.9.25 and the pinned PyTorch 1.12.1+cu113 dependencies.
Set the two dataset paths below to the directories described in [Data](#data).
The model files stay outside the image; results are saved to the host's `outputs/`.

```bash
RADIAL_DATA=/path/to/RADIal
RADDET_DATA=/path/to/RADDet
A2_PROJECT="$PWD"
mkdir -p outputs

a2run() {
  docker run --rm \
    --gpus device=0 \
    --user "$(id -u):$(id -g)" \
    --shm-size=2g \
    --mount "type=bind,src=$RADIAL_DATA,dst=/data/RADIal,readonly" \
    --mount "type=bind,src=$RADDET_DATA,dst=/data/RADDet,readonly" \
    --mount "type=bind,src=$A2_PROJECT/checkpoints,dst=/checkpoints,readonly" \
    --mount "type=bind,src=$A2_PROJECT/outputs,dst=/workspace/app/outputs" \
    a2radnet:verify "$@"
}
```

Evaluate the supplied checkpoints:

```bash
a2run python evaluate.py --config configs/radial.json \
  --data-root /data/RADIal --checkpoint /checkpoints/radial_seed3.pth \
  --output outputs/radial_pretrained/results.json

a2run python evaluate.py --config configs/raddet.json \
  --data-root /data/RADDet --checkpoint /checkpoints/raddet_seed3.pth \
  --output outputs/raddet_pretrained/results.json
```

Train a single seed:

```bash
a2run python train.py --config configs/radial.json \
  --data-root /data/RADIal --seed 3 --output-dir outputs/radial

a2run python train.py --config configs/raddet.json \
  --data-root /data/RADDet --seed 3 --output-dir outputs/raddet
```

For a quick training check, add `--smoke-batches 1`.
For data verification, use `scripts/check_data.py` with the appropriate config and
`/data/RADIal` or `/data/RADDet` as `--data-root`.

If your network requires a proxy, pass your existing proxy environment at build
time. Keep network-specific settings outside the Dockerfile:

```bash
docker build --build-arg http_proxy --build-arg https_proxy \
  --build-arg no_proxy -t a2radnet:verify .
```

## Data

Obtain data from the dataset owners; this repository redistributes no dataset.

- [RADIal / FFTRadNet](https://github.com/valeoai/RADIal): use the ready-to-use
  complex `radar_FFT`, freespace masks, camera images, and `labels.csv`.
- [RADDet](https://github.com/ZhangAoCanada/RADDet): use the ADC NumPy files and
  the official training/test ground-truth boxes. Range/Doppler FFT preprocessing
  runs in the data loader; angle FFT/RDA files are unnecessary.

```text
RADIal/
├── labels.csv
├── radar_FFT/       # fft_000000.npy, complex [512,256,16]
├── radar_Freespace/ # freespace_000000.png
└── camera/          # image_000000.jpg

RADDet/
├── ADC/             # 000000.npy, reshaped to [256,64,8]
├── gt_box/          # official training annotations, 000000.pickle
└── gt_box_test/     # official test annotations
```

Verify the installed annotations, exact split IDs and required files against the
recorded seed-3 dataset before running:

```bash
python scripts/check_data.py --config configs/radial.json --data-root /path/to/RADIal
python scripts/check_data.py --config configs/raddet.json --data-root /path/to/RADDet
```

This checks annotation content hashes and required file names, not the full radar
array contents. Pass dataset locations with `--data-root`. No link to a parent research repository
or external FFTRadNet checkout is needed. Normalization values in the supplied
configs are preserved from the recorded runs; do not recompute them on test data.
The maintainer confirms that normalization statistics were computed from training
data only; validation and test data were not used.

## Evaluate pretrained models

Two model-only seed-3 checkpoints are included under `checkpoints/`. They contain
no optimizer state or source paths; adjacent JSON files record hashes and epochs.

| Dataset | Model checkpoint | Size | Metadata |
|---|---|---|---|
| RADIal | [radial_seed3.pth](../checkpoints/radial_seed3.pth) | 38.68 MiB | [manifest](../checkpoints/radial_seed3.json) |
| RADDet | [raddet_seed3.pth](../checkpoints/raddet_seed3.pth) | 28.98 MiB | [manifest](../checkpoints/raddet_seed3.json) |

Only these two checkpoints and their manifests are included in Git. Training
outputs and other checkpoints remain excluded. See [licensing details](licensing.md)
for the recorded code and model-weight terms.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/radial.json \
  --data-root /path/to/RADIal --checkpoint checkpoints/radial_seed3.pth \
  --output outputs/radial_pretrained/results.json

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/raddet.json \
  --data-root /path/to/RADDet --checkpoint checkpoints/raddet_seed3.pth \
  --output outputs/raddet_pretrained/results.json
```

Evaluation strictly checks every model key and tensor shape. Results are accompanied
by metadata (seed, epoch, checkpoint hash, effective config, environment) and exact
train/validation/test sample IDs. Only load trusted pickle-based checkpoints.

`reference_results/*_seed3.json` contains the historical single-seed results used
by the existing table scripts. These are reference values, not three-seed paper
averages. See [validation status](validation.md) for freshly verified results.

## Train and evaluate one seed

The training code includes only the three task losses used by each dataset.
Auxiliary map heads/losses and unused point-cloud payloads have been removed.
Freespace segmentation on RADIal and class prediction on RADDet are retained.
Existing pretrained model tensors and their parameter names are unchanged.


```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/radial.json \
  --data-root /path/to/RADIal --seed 3 --output-dir outputs/radial

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/raddet.json \
  --data-root /path/to/RADDet --seed 3 --output-dir outputs/raddet

python evaluate.py --run outputs/radial/RUN_NAME --data-root /path/to/RADIal \
  --output outputs/radial/RUN_NAME/results.json

python evaluate.py --run outputs/raddet/RUN_NAME --data-root /path/to/RADDet \
  --output outputs/raddet/RUN_NAME/results.json
```

Training defaults to the recorded 150 epochs, Adam (learning rate 0.0001),
StepLR (0.9 every 10 epochs), and batch sizes 4/16 for RADIal/RADDet. An explicit
`--checkpoint` can be supplied with `--run` to evaluate a chosen epoch.
Automatic selection uses validation F1 only, with the first 20 epochs excluded
for RADIal. On the full 150-epoch histories this selects RADIal epoch 118 and
RADDet epoch 141, matching the historical evaluation files. Epoch numbers are
zero based. The selector identifies the final checkpoint by numeric epoch.

Use `--resume path/to/checkpoint.pth` with the full training checkpoint to resume
optimizer/scheduler/history in a new run folder. Model-only exports cannot resume
training. Resume does not promise bitwise equivalence to an uninterrupted run.

For a quick real-data forward/backward/validation/checkpoint check:

```bash
python train.py --config configs/radial.json --data-root /path/to/RADIal --smoke-batches 1
python train.py --config configs/raddet.json --data-root /path/to/RADDet --smoke-batches 1
```

Smoke runs use one epoch and limited data, have `_smoke` in their names, and are
not trained reproduction models. `--epochs` changes the total epoch count;
`--num-workers` overrides all data-loader worker counts. The default is 0 because
forked MKL FFT workers crashed during validation on the inspected environment.

## Tables, checks, and model export

```bash
python scripts/summarize_results.py \
  --radial outputs/radial_pretrained/results.json \
  --raddet outputs/raddet_pretrained/results.json --output outputs/summary.csv

python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/compare_results.py --reference reference_results/radial_seed3.json \
  --actual outputs/radial_pretrained/results.json

python scripts/export_checkpoint.py --config configs/radial.json \
  --checkpoint /path/to/training_checkpoint.pth --output checkpoints/radial_seed3.pth
```

The result summarizer reports explicit single-seed rows and fails on missing input
files rather than silently reducing the number of evaluated seeds. It distinguishes
binary detection precision from class-wise AP. [Reproduction notes](reproduction.md)
describe the inherited metric definitions.

## Model and paper terminology

| Paper component | Implementation | RADIal | RADDet |
|---|---|---|---|
| A²-RadNet | `a2radnet/models/{radial,raddet}.py:A2RadNet` | Detection + freespace | Detection + classification |
| Angle-Aware Encoder (AAE) | `a2radnet/models/angle_aware_encoder.py:AngleAwareEncoder` | Doppler-Structured | Regular-Antenna |
| Doppler-Structured projection | `DilatedCircularProjection` | Circular convolution, dilation 16 | — |
| Regular-Antenna projection | `PointwiseProjection` | — | Two pointwise convolutions |
| Doppler Latent Reducer (DLR) | `DopplerLatentReducer` | 32 latent tokens | — |
| RPA backbone | `a2radnet/models/swin.py:SwinTransformer` | Spatial Swin | Spatial Swin |
| Multi-scale fusion | `RangePseudoAngleDecoder` | RPA feature pyramid | RPA feature pyramid |

The channel-first interface is `[B, 2C, R, D] → [B, L, R, Ap]`, with real channels
followed by imaginary channels. Pseudo-angle bins are learned feature indices,
not calibrated azimuth bins. Existing state-dict names such as `frontend`, `vit`
and `RA_decoder` are retained for checkpoint compatibility. The historical
`data_mode="RA"` selects the proposed method; its **input is complex RD**.

## Prepare a source archive

```bash
python scripts/package_release.py --output dist/A2RadNet-source.tar.gz
# Optional local bundle containing sanitized weights:
python scripts/package_release.py --include-weights --output dist/A2RadNet-with-weights.tar.gz
```

The archive uses an explicit source allowlist, excludes runtime outputs and parent
Git history, and removes local owner/time metadata from tar entries. Resolve the
licensing and reproduction notes before public distribution. For a new Git
repository, copy this folder's source and respect `.gitignore`; keep runtime
`outputs/` private. Only the two listed model checkpoints and manifests are
included in the source repository.

## Attribution and citation

Original A²-RadNet code and the provided pretrained models are available under
[ITLAB non-commercial terms](../LICENSE). If you use either in your research,
please cite the paper using the [BibTeX in the README](../README.md#citation).


This work builds on [T-FFTRadNet](https://github.com/jgiroux8/T_FFTRadNet), which in
turn builds on [FFTRadNet / RADIal](https://github.com/valeoai/RADIal). The Swin
backbone retains the Microsoft copyright and [MIT license](../LICENSES/Swin-Transformer-MIT.txt).
See [NOTICE.txt](../NOTICE.txt), [licensing status](licensing.md), and
[CITATION.cff](../CITATION.cff). Paper DOI and proceedings pages should be added once available.
