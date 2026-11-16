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

class RADDet(Dataset):

    def __init__(self, root_dir, label_encoder, mode='Total', statistics=None, encoder=None, perform_FFT=False, angle_bins=128):
        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        self.angle_bins = angle_bins
        if not os.path.exists(root_dir):
            raise FileNotFoundError('Dataset root does not exist: ' + str(root_dir))
        self.index_path = os.path.join(self.root_dir, 'ADC')
        self.mode = mode
        if self.mode == 'Train':
            self.FFT_label_idx = sorted([int(e.replace('.pickle', '')) for e in os.listdir(os.path.join(self.root_dir, 'gt_box'))])
            self.label_dir = os.path.join(self.root_dir, 'gt_box')
        elif self.mode == 'Test':
            self.FFT_label_idx = sorted([int(e.replace('.pickle', '')) for e in os.listdir(os.path.join(self.root_dir, 'gt_box_test'))])
            self.label_dir = os.path.join(self.root_dir, 'gt_box_test')
        self.numSamplePerChirp = 256
        self.numChirps = 64
        self.numRxAnt = 8
        self.numTxAnt = 2
        hanningWindowRange = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.numSamplePerChirp) / (self.numSamplePerChirp - 1))
        hanningWindowDoppler = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(self.numChirps) / (self.numChirps - 1))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange, 1), repeats=self.numChirps, axis=1), 2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0), 2)
        self.perform_FFT = perform_FFT
        self.label_encoder = label_encoder
        self.range_resolution = 0.1953125
        self.angle_resolution = 0.703125

    def __len__(self):
        return len(self.FFT_label_idx)

    def __getitem__(self, index):
        sample_id = self.FFT_label_idx[index]
        gt_name = os.path.join(self.label_dir, '{:06d}.pickle'.format(sample_id))
        gt_box = np.load(gt_name, allow_pickle=True)
        box_labels = []
        for i in range(len(gt_box['cart_boxes'])):
            if i > 30:
                break
            class_ = [gt_box['classes'][i]]
            class_ = self.label_encoder.transform(class_)[0]
            y_center = 50 - gt_box['cart_boxes'][i][0] * 50 / 256
            x_center = gt_box['cart_boxes'][i][1] * 100 / 512 - 50.0
            R = np.sqrt(x_center ** 2 + y_center ** 2)
            phi = np.degrees(np.arctan(x_center / y_center))
            box_labels.append([R, phi, class_])
        out_label = []
        if self.encoder != None:
            out_label = self.encoder(box_labels).copy()
            class_map = out_label[3:]
            out_label = out_label[:3]
        radar_name = os.path.join(self.root_dir, 'ADC', '{:06d}.npy'.format(sample_id))
        if self.perform_FFT == 'ADC':
            complex_adc = np.load(radar_name).reshape(256, 64, 8)
            radar_FFT = complex_adc
        elif self.perform_FFT == 'RD' or self.perform_FFT == 'RA':
            complex_adc = np.load(radar_name).reshape(256, 64, 8)
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0, 1))
            range_fft = mkl_fft.fft(np.multiply(complex_adc, self.range_fft_coef), self.numSamplePerChirp, axis=0)
            input = mkl_fft.fft(np.multiply(range_fft, self.doppler_fft_coef), self.numChirps, axis=1)
            input = np.fft.fftshift(input, axes=1)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)
            if self.statistics is not None:
                if 'normalization' in self.statistics.keys() and self.statistics['normalization'] == 'global_power_RMS':
                    radar_FFT = radar_FFT / self.statistics['input_global_scale']
                else:
                    for i in range(len(self.statistics['input_mean'])):
                        radar_FFT[..., i] -= self.statistics['input_mean'][i]
                        radar_FFT[..., i] /= self.statistics['input_std'][i]
        else:
            RAD_dir = os.path.join(self.root_dir, 'RAD' if self.mode == 'Train' else 'RAD_test')
            radar_name = os.path.join(RAD_dir, '{:06d}.npy'.format(sample_id))
            radar_FFT = np.load(radar_name)
            radar_FFT = np.flip(np.abs(radar_FFT), axis=0)
        return (radar_FFT, out_label, np.asarray(box_labels), class_map)
