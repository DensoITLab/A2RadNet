"""Export model tensors without optimizer state, paths or training logs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from a2radnet.models.factory import build_model
    from a2radnet.runtime import load_checkpoint, sha256, write_json
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output already exists; choose a new filename.')
    config = json.loads(args.config.read_text())
    checkpoint = load_checkpoint(args.checkpoint)
    state = checkpoint.get('net_state_dict', checkpoint)
    net = build_model(config)
    net.load_state_dict(state, strict=True)
    public = {'net_state_dict': {name: tensor.detach().cpu() for name, tensor in state.items()},
              'epoch': checkpoint.get('epoch'), 'seed': config['seed']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, args.output)
    write_json(args.output.with_suffix('.json'), {
        'dataset': config['dataset_name'], 'seed': config['seed'], 'epoch': checkpoint.get('epoch'),
        'sha256': sha256(args.output), 'source_checkpoint_sha256': sha256(args.checkpoint),
        'config_sha256': sha256(args.config), 'format': 'net_state_dict, epoch, seed',
        'license_status': 'Original A2-RadNet code and pretrained models: ITLAB license (non-commercial); see LICENSE'})
    print('Exported:', args.output)


if __name__ == '__main__':
    main()
