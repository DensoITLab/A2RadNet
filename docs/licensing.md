# Licensing

Original A²-RadNet code contributions and the two provided pretrained models are
licensed under the **ITLAB license for non-commercial use**. The full terms are
in [LICENSE](../LICENSE).

Copyright (C) 2026 Denso IT Laboratory, Inc. All Rights Reserved.

The supplied terms permit use, copying, modification, merging, publication and
distribution for non-commercial purposes, such as academic study, education and
personal use, subject to the notices and disclaimer in LICENSE. They do not grant
permission for commercial use. Source and binary redistribution must retain or
reproduce the required notices, conditions and disclaimer as specified there.

## Scope and third-party components

| Source | Use in this repository | Applicable terms |
|---|---|---|
| Original A²-RadNet contributions | AAE/RPA changes and release tooling | ITLAB license; non-commercial use. |
| Provided pretrained models | `radial_seed3.pth` and `raddet_seed3.pth` | ITLAB license; non-commercial use. |
| [T-FFTRadNet](https://github.com/jgiroux8/T_FFTRadNet) | Model/backbone adaptation, training and benchmark pipeline | Upstream provenance retained. No root license was found in the inspected checkout/public root listing; its applicable redistribution terms remain unverified. |
| [FFTRadNet / RADIal](https://github.com/valeoai/RADIal) | Inherited data handling, losses and evaluation through T-FFTRadNet | Upstream provenance retained. No root license was found in the inspected public root listing; nested/applicable terms remain unverified. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | Spatial backbone | Microsoft MIT attribution and full license retained in `LICENSES/Swin-Transformer-MIT.txt`. |
| [RADDet](https://github.com/ZhangAoCanada/RADDet) | Dataset/annotation format | Dataset is not bundled; its terms are separate. |

The ITLAB license applies to the original contributions and supplied model weights.
It does not replace third-party licenses or remove their copyright notices.
The ownership statement in LICENSE does not claim ownership of third-party material.
Attribution and adaptation details are in [NOTICE.txt](../NOTICE.txt).

## Citation

If you use this code or the pretrained models in your research, please cite
“Learning Structured Angle-Aware Representations for Radar Object Detection”
by Jun Nagata, Tatsuya Ueda and Koichiro Suzuki, BMVC 2026.
BibTeX is provided in the [README](../README.md#citation), and machine-readable
citation metadata is in [CITATION.cff](../CITATION.cff).

This is an academic citation request. The supplied license text has not been
extended with an additional mandatory-citation condition.
