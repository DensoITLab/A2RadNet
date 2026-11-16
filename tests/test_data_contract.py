"""Keep task tensors in their original slots after removing unused batch fields."""
import unittest

import numpy as np
import torch
from a2radnet.radial.dataloader import RADIal_collate as radial_collate
from a2radnet.raddet.dataloader import RADIal_collate as raddet_collate


class TaskBatchTests(unittest.TestCase):
    def test_radial_batch_keeps_detection_freespace_labels_and_images(self):
        radar = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        segmentation = np.ones((2, 3), dtype=bool)
        detection = np.zeros((3, 2, 3), dtype=np.float32)
        boxes = np.array([[10.0, 2.0]], dtype=np.float32)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        batch = radial_collate([(radar, segmentation, detection, boxes, image)] * 2)
        self.assertEqual(len(batch), 5)
        torch.testing.assert_close(batch[0][0], torch.from_numpy(radar).permute(2, 0, 1))
        torch.testing.assert_close(batch[1][0], torch.from_numpy(detection))
        torch.testing.assert_close(batch[2][0], torch.from_numpy(segmentation))
        torch.testing.assert_close(batch[3][0], torch.from_numpy(boxes))
        torch.testing.assert_close(batch[4][0], torch.from_numpy(image))

    def test_raddet_batch_keeps_detection_boxes_and_class_targets(self):
        radar = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        detection = np.zeros((3, 2, 3), dtype=np.float32)
        boxes = np.array([[10.0, 2.0, 1.0]], dtype=np.float32)
        classes = np.zeros((6, 2, 3), dtype=np.float32)
        batch = raddet_collate([(radar, detection, boxes, classes)] * 2)
        self.assertEqual(len(batch), 4)
        torch.testing.assert_close(batch[0][0], torch.from_numpy(radar).permute(2, 0, 1))
        torch.testing.assert_close(batch[1][0], torch.from_numpy(detection))
        torch.testing.assert_close(batch[2][0], torch.from_numpy(boxes))
        torch.testing.assert_close(batch[3][0], torch.from_numpy(classes))


if __name__ == '__main__':
    unittest.main()
