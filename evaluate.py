"""Evaluate a specified checkpoint on the official test split."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path)
    source.add_argument('--run', type=Path, help='Training run containing config.json and model/*.pth.')
    parser.add_argument('--checkpoint', type=Path, help='Explicit checkpoint; required with --config.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--num-workers', type=int, help='Override workers for all splits; 0 avoids fork/MKL issues.')
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation/results.json'))
    args = parser.parse_args()
    if args.config and not args.checkpoint:
        parser.error('--config requires --checkpoint')
    config_path = args.config or args.run / 'config.json'
    config = json.loads(config_path.read_text())
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error('--num-workers must be non-negative')
        for split in ['train', 'val', 'test']:
            config['dataloader'][split]['num_workers'] = args.num_workers
    if args.data_root:
        config['dataset']['root_dir'] = str(args.data_root.resolve())
    import torch
    from a2radnet.models.factory import build_model
    from a2radnet.runtime import (build_data, environment, load_checkpoint, seed_everything,
                                 select_checkpoint, sha256, split_manifest, write_json)
    if not torch.cuda.is_available():
        parser.error('This reference evaluation implementation requires a CUDA GPU.')
    checkpoint_path = args.checkpoint or select_checkpoint(args.run, config['dataset_name'])
    seed_everything(config['seed'])
    net = build_model(config)
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get('net_state_dict', checkpoint)
    net.load_state_dict(state, strict=True)
    net.cuda().eval()
    encoder, loaders = build_data(config)
    splits = split_manifest(loaders)
    evaluation = importlib.import_module('a2radnet.' + config['dataset_name'] + '.evaluation')
    if config['dataset_name'] == 'radial':
        evaluation.run_FullEvaluation(net, loaders[2], encoder, config=config,
                                      save_path=str(args.output), epoch=checkpoint.get('epoch'))
    else:
        metrics = importlib.import_module('a2radnet.raddet.metrics')
        results = metrics.run_APEvaluation(net, loaders[2], encoder, iou_threshold=0.5, config=config)
        results.update(evaluation.run_FullEvaluation(net, loaders[2], encoder, config=config))
        results['epoch'] = checkpoint.get('epoch')
        write_json(args.output, results)
    write_json(args.output.with_suffix('.metadata.json'), {
        'dataset': config['dataset_name'], 'seed': config['seed'],
        'checkpoint_sha256': sha256(checkpoint_path), 'epoch': checkpoint.get('epoch'),
        'config': config, 'environment': environment(),
        'split_counts': {name: len(ids) for name, ids in splits.items()}})
    write_json(args.output.with_suffix('.splits.json'), splits)
    print('Evaluation saved:', args.output)


if __name__ == '__main__':
    main()
