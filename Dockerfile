FROM python:3.9.25-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

RUN python -m pip install --no-cache-dir \
    pip==24.1.2 setuptools==71.1.0 wheel==0.46.3

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY pyproject.toml README.md LICENSE NOTICE.txt CITATION.cff ./
COPY a2radnet/ ./a2radnet/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY reference_results/ ./reference_results/
COPY docs/ ./docs/
COPY LICENSES/ ./LICENSES/
COPY train.py evaluate.py ./

RUN python -m pip install --no-deps --no-build-isolation . \
    && python -m unittest discover -s tests -v

CMD ["python", "--version"]
