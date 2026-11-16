"""Shared execution metadata and deterministic checkpoint selection."""
import hashlib
import importlib
from importlib import metadata
import json
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'timm', 'numpy', 'scipy', 'pandas', 'mkl_fft',
                 'scikit-learn', 'shapely', 'Pillow', 'pkbar', 'tensorboard']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'torch_cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    # Research checkpoints are pickle-based; load only trusted model files.
    return torch.load(path, map_location='cpu')


def select_checkpoint(run, dataset, ignore_epochs=None):
    files = list((Path(run) / 'model').glob('*.pth'))
    def epoch(path):
        match = re.search(r'_epoch(\d+)_', path.name)
        if not match:
            raise ValueError('Checkpoint filename has no epoch: ' + path.name)
        return int(match.group(1))
    if not files:
        raise FileNotFoundError('No model/*.pth checkpoints in ' + str(run))
    latest = max(files, key=epoch)
    history = load_checkpoint(latest)['history']
    scores = [2*p*r/(p+r) if p+r else 0 for p, r in zip(history['mAP'], history['mAR'])]
    ignore = (20 if dataset == 'radial' else 0) if ignore_epochs is None else ignore_epochs
    if not 0 <= ignore < len(scores):
        raise ValueError('Not enough epochs for checkpoint selection; supply --checkpoint for smoke runs.')
    best = max(range(ignore, len(scores)), key=lambda i: scores[i])
    matches = [p for p in files if epoch(p) == best]
    if len(matches) != 1:
        raise ValueError('Expected exactly one checkpoint for epoch ' + str(best))
    return matches[0]


def build_data(config):
    dataset = config['dataset_name']
    module = importlib.import_module('a2radnet.' + dataset + '.dataset')
    encoder_class = importlib.import_module('a2radnet.' + dataset + '.encoder').ra_encoder
    create = importlib.import_module('a2radnet.' + dataset + '.dataloader').CreateDataLoaders
    kwargs = dict(geometry=config['dataset']['geometry'], statistics=config['dataset']['statistics'], regression_layer=2)
    common = dict(root_dir=config['dataset']['root_dir'], statistics=config['dataset']['statistics'], perform_FFT=config['data_mode'])
    if dataset == 'radial':
        encoder = encoder_class(**kwargs, label_map=config['label_map'])
        data = module.RADIal(**common, encoder=encoder.encode, difficult=True, cfg=config)
        loaders = create(data, config['dataloader'], config['seed'])
    else:
        from sklearn.preprocessing import LabelEncoder
        labels = LabelEncoder().fit(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])
        encoder = encoder_class(**kwargs)
        common.update(label_encoder=labels, encoder=encoder.encode,
                      angle_bins=config['dataset'].get('FFT_angle_bins', 128))
        data = module.RADDet(**common, mode='Train')
        test = module.RADDet(**common, mode='Test')
        loaders = create(data, test, config['dataloader'], config['seed'])
    return encoder, loaders


def split_manifest(loaders):
    splits = {}
    for name, loader in zip(['train', 'val', 'test'], loaders):
        dataset = loader.dataset
        indices = list(dataset.indices) if hasattr(dataset, 'indices') else list(range(len(dataset)))
        base = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        keys = base.sample_keys if hasattr(base, 'sample_keys') else base.FFT_label_idx
        splits[name] = [int(keys[int(i)]) for i in indices]
    sets = [set(splits[name]) for name in ['train', 'val', 'test']]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Dataset split contains overlapping sample IDs.')
    return splits


def prepare_training_loaders(loaders, config, run):
    # Record full split membership before limiting smoke runs.
    write_json(Path(run) / 'splits.json', split_manifest(loaders))
    write_json(Path(run) / 'environment.json', environment())
    count = config.get('smoke_batches', 0)
    if not count:
        return loaders
    from torch.utils.data import DataLoader, Subset
    return tuple(DataLoader(Subset(loader.dataset, range(min(len(loader.dataset), count * loader.batch_size))),
                            batch_size=loader.batch_size, shuffle=(i == 0), num_workers=0,
                            collate_fn=loader.collate_fn) for i, loader in enumerate(loaders))
