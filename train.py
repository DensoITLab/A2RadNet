"""Train one seed of the proposed A²-RadNet model."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--num-workers', type=int, help='Override workers for all splits; 0 avoids fork/MKL issues.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-batches', type=int, default=0,
                        help='Run one epoch on this many batches per split; not a reproduction result.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error('--num-workers must be non-negative')
        for split in ['train', 'val', 'test']:
            config['dataloader'][split]['num_workers'] = args.num_workers
    if args.data_root:
        config['dataset']['root_dir'] = str(args.data_root.resolve())
    if args.output_dir:
        config['output']['dir'] = str(args.output_dir.resolve())
    if args.seed is not None:
        config['seed'] = args.seed
        config['name'] = 'a2radnet_' + config['dataset_name'] + '_seed' + str(args.seed)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error('--epochs must be positive')
        config['num_epochs'] = args.epochs
    if args.smoke_batches < 0:
        parser.error('--smoke-batches must be non-negative')
    if args.smoke_batches:
        config['num_epochs'] = 1
        config['smoke_batches'] = args.smoke_batches
        config['name'] += '_smoke'
    if not Path(config['dataset']['root_dir']).is_dir():
        parser.error('Dataset root does not exist; use --data-root.')
    import torch
    if not torch.cuda.is_available():
        parser.error('This reference training implementation requires a CUDA GPU.')
    from a2radnet.models.factory import build_model
    # Validate config before creating output directories.
    del_model = build_model(config)
    del del_model
    Path(config['output']['dir']).mkdir(parents=True, exist_ok=True)
    training = importlib.import_module('a2radnet.' + config['dataset_name'] + '.training')
    run = training.main(config, str(args.resume) if args.resume else None)
    print('Training run:', run)


if __name__ == '__main__':
    main()
