# A²-RadNet

**Learning Structured Angle-Aware Representations for Radar Object Detection**  
Jun Nagata, Tatsuya Ueda, Koichiro Suzuki · DENSO IT Laboratory · BMVC 2026

A²-RadNet learns Range–Pseudo-Angle (RPA) features from complex range–Doppler
inputs before spatial backbone processing. This repository provides the proposed
method and pretrained models for RADIal and RADDet.

## Setup

Use Python 3.9 and a CUDA-capable GPU. PyTorch 1.12.1+cu113 and the other runtime
dependencies are pinned in `requirements.txt`.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

For container-based setup, follow the [Docker instructions](docs/usage.md#docker).
All commands below run from the repository root, in the configured Python environment.

## Data

Download data from [RADIal](https://github.com/valeoai/RADIal) and
[RADDet](https://github.com/ZhangAoCanada/RADDet).
RADIal uses the ready-to-use complex RD data; RADDet uses ADC data and annotations.
See the [expected directory layouts](docs/usage.md#data).

## Evaluate pretrained models

| Dataset | Config | Checkpoint |
|---|---|---|
| RADIal | [radial.json](configs/radial.json) | [radial_seed3.pth](checkpoints/radial_seed3.pth) |
| RADDet | [raddet.json](configs/raddet.json) | [raddet_seed3.pth](checkpoints/raddet_seed3.pth) |

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/radial.json \
  --data-root /path/to/RADIal --checkpoint checkpoints/radial_seed3.pth \
  --output outputs/radial/results.json

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/raddet.json \
  --data-root /path/to/RADDet --checkpoint checkpoints/raddet_seed3.pth \
  --output outputs/raddet/results.json
```

These checkpoints are from seed 3. The paper reports three-seed averages;
see [reproduction details](docs/reproduction.md) and [validation results](docs/validation.md).

## Train

The supplied configs train for 150 epochs with seed 3.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/radial.json \
  --data-root /path/to/RADIal --seed 3 --output-dir outputs/radial

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/raddet.json \
  --data-root /path/to/RADDet --seed 3 --output-dir outputs/raddet
```

For evaluation of trained runs, resume, smoke tests, and result summaries, see
[detailed usage](docs/usage.md).

## Citation

If you use this code or the pretrained models in your research, please cite our paper:

```bibtex
@inproceedings{nagata2026a2radnet,
  title     = {Learning Structured Angle-Aware Representations for Radar Object Detection},
  author    = {Nagata, Jun and Ueda, Tatsuya and Suzuki, Koichiro},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

Original A²-RadNet code and the provided pretrained models are available under the
[ITLAB license](LICENSE) for **non-commercial use**.
Third-party components retain their own licenses; see [NOTICE](NOTICE.txt)
and [licensing details](docs/licensing.md).

This implementation builds on [T-FFTRadNet](https://github.com/jgiroux8/T_FFTRadNet),
[FFTRadNet/RADIal](https://github.com/valeoai/RADIal), and Swin Transformer.
