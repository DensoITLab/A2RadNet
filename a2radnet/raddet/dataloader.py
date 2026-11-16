import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import torch

def RADIal_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []
    class_maps = []
    for (radar_FFT, out_label, box_labels, class_map) in batch:
        FFTs.append(torch.tensor(radar_FFT.copy()).permute(2, 0, 1))
        encoded_label.append(torch.tensor(out_label))
        labels.append(torch.from_numpy(box_labels))
        class_maps.append(torch.tensor(class_map))
    return (torch.stack(FFTs), torch.stack(encoded_label), labels, torch.stack(class_maps))

def CreateDataLoaders(dataset, test_dataset, config=None, seed=0):
    if config['mode'] == 'random':
        n_images = dataset.__len__()
        split = np.array(config['split'])
        if np.sum(split) != 1.0:
            raise NameError('The sum of the train/val split should be equal to 1')
            return
        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        if n_train + n_val != n_images:
            n_val += 1
        (train_dataset, val_dataset) = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Train Val ratio:', config['split'])
        print('      Training:', len(train_dataset), ' indexes...', train_dataset.indices[:3])
        print('      Validation:', len(val_dataset), ' indexes...', val_dataset.indices[:3])
        print('      Test:', len(test_dataset))
        print('')
        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=config['val']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False, num_workers=config['test']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        return (train_loader, val_loader, test_loader)
    else:
        raise ValueError('RADDet release supports the random train/validation split only')
