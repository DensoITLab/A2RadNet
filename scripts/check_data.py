"""Verify annotation fingerprints, sample splits and required dataset files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def annotations_sha256(root, dataset):
    paths = [root / 'labels.csv'] if dataset == 'radial' else sorted(
        list((root / 'gt_box').glob('*.pickle')) + list((root / 'gt_box_test').glob('*.pickle')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {'file_count': len(paths), 'sha256': digest.hexdigest(),
            'scheme': 'SHA256 of sorted relative POSIX names + NUL + SHA256 bytes per annotation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['dataset']['root_dir'] = str(args.data_root.resolve())
    from a2radnet.runtime import build_data, split_manifest, seed_everything
    seed_everything(config['seed'])
    _, loaders = build_data(config)
    splits = split_manifest(loaders)
    dataset = config['dataset_name']
    repository = Path(__file__).resolve().parents[1]
    expected = json.loads((repository / 'reference_results' / f'{dataset}_seed3.splits.json').read_text())
    if splits != expected:
        raise SystemExit('FAIL: sample IDs differ from the recorded seed-3 split.')
    fingerprints = json.loads((repository / 'reference_results' / 'dataset_fingerprints.json').read_text())
    if annotations_sha256(args.data_root.resolve(), dataset) != fingerprints[dataset]:
        raise SystemExit('FAIL: annotation fingerprints differ from the validated dataset.')
    ids = sorted({sample for split in splits.values() for sample in split})
    missing = []
    for sample in ids:
        relative = ([f'radar_FFT/fft_{sample:06d}.npy', f'radar_Freespace/freespace_{sample:06d}.png',
                     f'camera/image_{sample:06d}.jpg'] if dataset == 'radial' else [f'ADC/{sample:06d}.npy'])
        missing.extend(name for name in relative if not (args.data_root / name).is_file())
    if missing:
        raise SystemExit(f'FAIL: {len(missing)} required files missing; first: {missing[0]}')
    print('PASS: annotation fingerprints, exact split membership and required files.')
    print('Radar file contents are not hashed by this check.')


if __name__ == '__main__':
    main()
