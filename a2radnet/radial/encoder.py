import numpy as np

class ra_encoder:

    def __init__(self, geometry, statistics, regression_layer=2, label_map={'detection': [128, 112], 'segmentation': [256, 224]}):
        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer
        self.label_map = label_map
        self.factor_r = 512 // label_map['detection'][0]
        self.factor_a = 448 // label_map['detection'][1]
        self.INPUT_DIM = (geometry['ranges'][0], geometry['ranges'][1], geometry['ranges'][2])
        self.OUTPUT_DIM = (regression_layer + 1, self.INPUT_DIM[0] // self.factor_r, self.INPUT_DIM[1] // self.factor_a)
        self.range_res = self.geometry['resolution'][0] * self.factor_r
        self.angle_res = self.geometry['resolution'][1] * self.factor_a
        self.angle_center = self.OUTPUT_DIM[2] / 2

    def encode(self, labels):
        (H, W) = (self.OUTPUT_DIM[1], self.OUTPUT_DIM[2])
        map = np.zeros(self.OUTPUT_DIM)
        for lab in labels:
            if lab[0] == -1:
                continue
            range_bin = int(np.clip(lab[0] / self.range_res, 0, H - 1))
            range_mod = lab[0] - range_bin * self.range_res
            angle_bin = int(np.clip(np.floor(self.angle_center - lab[1] / self.angle_res), 0, W - 1))
            angle_mod = lab[1] - (self.angle_center - angle_bin) * self.angle_res
            if self.geometry['size'] == 1:
                map[0, range_bin, angle_bin] = 1
                map[1, range_bin, angle_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2, range_bin, angle_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
            else:
                s = int((self.geometry['size'] - 1) / 2)
                r_lin = np.linspace(self.geometry['resolution'][0] * s, -self.geometry['resolution'][0] * s, self.geometry['size']) * self.factor_r
                a_lin = np.linspace(-self.geometry['resolution'][1] * s, +self.geometry['resolution'][1] * s, self.geometry['size']) * self.factor_a
                (px_a, px_r) = np.meshgrid(a_lin, r_lin)
                if angle_bin >= s and angle_bin < W - s:
                    map[0, range_bin - s:range_bin + (s + 1), angle_bin - s:angle_bin + (s + 1)] = 1
                    map[1, range_bin - s:range_bin + (s + 1), angle_bin - s:angle_bin + (s + 1)] = (px_r + range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2, range_bin - s:range_bin + (s + 1), angle_bin - s:angle_bin + (s + 1)] = (px_a + angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                elif angle_bin < s:
                    map[0, range_bin - s:range_bin + (s + 1), 0:angle_bin + (s + 1)] = 1
                    map[1, range_bin - s:range_bin + (s + 1), 0:angle_bin + (s + 1)] = (px_r[:, s - angle_bin:] + range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2, range_bin - s:range_bin + (s + 1), 0:angle_bin + (s + 1)] = (px_a[:, s - angle_bin:] + angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                elif angle_bin >= W - s:
                    end = s + (W - angle_bin)
                    map[0, range_bin - s:range_bin + (s + 1), angle_bin - s:] = 1
                    map[1, range_bin - s:range_bin + (s + 1), angle_bin - s:] = (px_r[:, :end] + range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2, range_bin - s:range_bin + (s + 1), angle_bin - s:] = (px_a[:, :end] + angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
        return map

    def decode(self, map, threshold):
        (range_bins, angle_bins) = np.where(map[0, :, :] >= threshold)
        coordinates = []
        for (range_bin, angle_bin) in zip(range_bins, angle_bins):
            R = range_bin * self.range_res + map[1, range_bin, angle_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]
            A = (self.angle_center - angle_bin) * self.angle_res + map[2, range_bin, angle_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]
            C = map[0, range_bin, angle_bin]
            coordinates.append([R, A, C])
        return coordinates
