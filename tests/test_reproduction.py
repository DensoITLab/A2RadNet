"""Contract tests for RPA layout, periodic gathering and checkpoint selection."""
import tempfile
import unittest
from pathlib import Path

import torch
from a2radnet.models.angle_aware_encoder import (AngleAwareEncoder, DopplerLatentReducer,
                                                cyclic_padding)
from a2radnet.runtime import select_checkpoint


class RepresentationTests(unittest.TestCase):
    def test_circular_padding_preserves_periodic_neighbors(self):
        x = torch.arange(8).reshape(1, 1, 1, 8)
        self.assertEqual(cyclic_padding(x, 3).flatten().tolist(), [5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7])

    def test_dlr_aggregates_unordered_doppler_hypotheses(self):
        torch.manual_seed(3)
        reducer = DopplerLatentReducer(feat_dim=8, num_latents=3).eval()
        x = torch.randn(2, 4, 16, 8, requires_grad=True)
        first = reducer(x)
        second = reducer(x[:, :, torch.randperm(16), :])
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
        first.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(reducer.latents.grad).all())

    def test_both_aae_variants_produce_rpa_and_receive_gradients(self):
        for structured in [False, True]:
            encoder = AngleAwareEncoder(doppler_bins=16, doppler_period=4,
                                         num_ant=2, out_channels=8, num_latents=3,
                                         doppler_reducer=int(structured),
                                         use_cyclic_shift=structured, output_order='ra').eval()
            x = torch.randn(1, 4, 5, 16, requires_grad=True)
            out = encoder(x)
            self.assertEqual(tuple(out.shape), (1, 3 if structured else 16, 5, 8))
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_checkpoint_selection_uses_numeric_epoch_and_validation_only(self):
        for dataset, best_epoch in [('radial', 118), ('raddet', 141)]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as folder:
                models = Path(folder) / 'model'
                models.mkdir()
                scores = [0.0] * 150
                scores[98] = 0.8
                scores[best_epoch] = 0.9
                # Write epoch 99 last: lexical ordering and change time are both
                # unsuitable ways to find the complete 150-epoch history.
                for epoch in [149, best_epoch, 99]:
                    history = scores[:epoch + 1]
                    torch.save({'history': {'mAP': history, 'mAR': history}},
                               models / f'run_epoch{epoch:02d}_loss_0.pth')
                selected = select_checkpoint(folder, dataset)
                self.assertEqual(selected.name, f'run_epoch{best_epoch}_loss_0.pth')


if __name__ == '__main__':
    unittest.main()
