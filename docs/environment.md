# Inspected execution environment

| Component | Value |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| Python used by pip | 3.9.5 |
| PyTorch | 1.12.1+cu113 |
| torchvision | 0.13.1+cu113 |
| PyTorch CUDA build | 11.3 |
| CUDA Toolkit (`nvcc`) | 11.3.109 |
| NVIDIA driver | 595.99.02 |
| GPU | NVIDIA RTX 6000 Ada Generation, about 48 GB each |
| Installed GPUs | 2; each supplied command uses a single GPU |
| timm | 1.0.26 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 2.3.3 |
| mkl_fft | 1.3.14 |
| scikit-learn | 1.6.1 |
| Shapely | 2.0.7 |
| Pillow | 11.3.0 |

On the inspected machine, `pip` invokes Python 3.9 while `python3` invokes Python
3.8. Use `python3.9 -m pip` or an activated Python 3.9 venv to avoid mixing them.
The pins describe the currently inspected runtime; historical training-time package
versions have not been established from saved logs.

`requirements.txt` pins direct runtime dependencies, not every transitive package
or system library. It is not yet a fully solved, independently tested lockfile.
A clean installation could not be tested because the package index was unreachable
from the execution environment. No existing system package was installed or updated.
The original Windows Conda export is not used by this repository.

Each real run records its actual Python, packages, CUDA/cuDNN, GPU, and cuDNN flags.
No hostnames, user account names or scheduler configuration are needed in this file.
Runtime GPU memory, full training duration and minimum hardware remain unmeasured.
