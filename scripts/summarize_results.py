"""Create proposed-method CSV rows from explicit evaluation JSON files."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radial', type=Path)
    parser.add_argument('--raddet', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.radial and not args.raddet:
        parser.error('Supply --radial and/or --raddet.')
    rows = []
    for dataset in ['radial', 'raddet']:
        path = getattr(args, dataset)
        if path is None:
            continue
        result = json.loads(path.read_text())
        scores = result['detection']['0.5']
        row = {'dataset': dataset, 'model': 'A2-RadNet', 'epoch': result['epoch'],
               'precision_at_iou_0.5': scores['Precision'], 'recall_at_iou_0.5': scores['Recall'],
               'range_error_m': scores['Range Error'], 'angle_error_deg': scores['Angle Error']}
        metadata = path.with_suffix('.metadata.json')
        if metadata.exists():
            row['seed'] = json.loads(metadata.read_text())['seed']
        if dataset == 'radial':
            row['freespace_mIoU'] = result['freespace']['mIoU']
        else:
            row['class_mAP'] = result['mAP']
            for name in ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']:
                row[name + '_AP'] = result[name]['AP']
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
