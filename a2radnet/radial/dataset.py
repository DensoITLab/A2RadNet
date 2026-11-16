from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize, CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import mkl_fft
import math

class RADIal(Dataset):

    def __init__(self, root_dir, statistics=None, encoder=None, difficult=True, perform_FFT='Default', cfg=None):
        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        self.perform_FFT = perform_FFT
        self.cfg = cfg
        if not os.path.exists(root_dir):
            raise FileNotFoundError('Dataset root does not exist: ' + str(root_dir))
        self.labels = pd.read_csv(os.path.join(self.root_dir, 'labels.csv')).to_numpy()
        self.numChirps = 256
        self.numSamplePerChirp = 512
        self.numRxAnt = 16
        self.numTxAnt = 12
        hanningWindowRange = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.numSamplePerChirp) / (self.numSamplePerChirp - 1))
        hanningWindowDoppler = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.numChirps) / (self.numChirps - 1))
        hanningWindowAzimuth = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.numRxAnt) / (self.numRxAnt - 1))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange, 1), repeats=self.numChirps, axis=1), 2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0), 2)
        if difficult == False:
            ids_filters = []
            ids = np.where(self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]
        self.unique_ids = np.unique(self.labels[:, 0])
        self.label_dict = {}
        for (i, ids) in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:, 0] == ids)[0]
            self.label_dict[ids] = sample_ids
        self.sample_keys = list(self.label_dict.keys())
        self.resize = Resize(cfg['label_map']['segmentation'], interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512, 448))

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        sample_id = self.sample_keys[index]
        entries_indexes = self.label_dict[sample_id]
        box_labels = self.labels[entries_indexes]
        box_labels = box_labels[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32)
        out_label = []
        if self.encoder != None:
            out_label = self.encoder(box_labels).copy()
        radar_name = os.path.join(self.root_dir, 'radar_FFT', 'fft_{:06d}.npy'.format(sample_id))
        input = np.load(radar_name, allow_pickle=True)
        radar_FFT = np.concatenate([input.real, input.imag], axis=2)
        if self.statistics is not None:
            if 'normalization' in self.statistics.keys() and self.statistics['normalization'] == 'global_power_RMS':
                radar_FFT = radar_FFT / self.statistics['input_global_scale']
            else:
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[..., i] -= self.statistics['input_mean'][i]
                    radar_FFT[..., i] /= self.statistics['input_std'][i]
        segmap_name = os.path.join(self.root_dir, 'radar_Freespace', 'freespace_{:06d}.png'.format(sample_id))
        segmap = Image.open(segmap_name)
        segmap = self.crop(segmap)
        segmap = np.asarray(self.resize(segmap)) == 255
        img_name = os.path.join(self.root_dir, 'camera', 'image_{:06d}.jpg'.format(sample_id))
        image = np.asarray(Image.open(img_name))
        return (radar_FFT, segmap, out_label, box_labels, image)
