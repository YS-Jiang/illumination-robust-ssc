import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SyntheticIlluminationDegradation(object):
    """Synthetic illumination degradation.

    Randomly pushes the color image into low-light / backlit / noisy regimes so the fusion
    receives gradient where the color SNR collapses. The grayscale image is degraded by a
    smaller factor, reflecting the higher SNR of a sensor without a color filter array.
    Ground truth and LiDAR depth are untouched (photometric only). The same operator at fixed
    strengths defines the graded evaluation protocol.
    """

    def __init__(self, prob=0.5, gamma_range=(1.6, 4.0), gain_range=(0.25, 0.8),
                 noise_std=0.04, backlight_prob=0.3, gray_strength=0.35,
                 degrade_depth=False, depth_noise_scale=0.35, depth_dropout=0.25,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.noise_std = noise_std
        self.backlight_prob = backlight_prob
        self.gray_strength = gray_strength   # gray degraded this fraction as hard as color (<1)
        # Diagnostic/realism: also corrupt the precomputed stereo depth (low-light stereo matching fails ->
        # noisier disparity + holes), severity scaled by darkness. Off by default (backward compatible).
        self.degrade_depth = degrade_depth
        self.depth_noise_scale = depth_noise_scale
        self.depth_dropout = depth_dropout
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def _unnorm(self, img):
        return (img * self.std + self.mean).clamp(0.0, 1.0)

    def _norm(self, img):
        return (img - self.mean) / self.std

    def _degrade(self, img01, gamma, gain, noise_std, backlight):
        # img01: (N,3,H,W) in [0,1]
        x = (img01.clamp_min(1e-6) ** gamma) * gain
        if backlight is not None:
            x = (x * backlight).clamp(0.0, 1.0)   # spatial gradient: bright sky, crushed foreground
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x.clamp(0.0, 1.0)

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        color = results['img_inputs'][0]            # (N,3,H,W) normalized
        N, C, H, W = color.shape
        gamma = float(np.random.uniform(*self.gamma_range))
        gain = float(np.random.uniform(*self.gain_range))
        backlight = None
        if np.random.rand() < self.backlight_prob:
            # vertical brightness gradient (top bright -> bottom dark), the backlight signature
            col = torch.linspace(1.0, 0.25, H).view(1, 1, H, 1)
            backlight = col.expand(N, 1, H, W)

        c01 = self._unnorm(color)
        c01 = self._degrade(c01, gamma, gain, self.noise_std, backlight)
        results['img_inputs'][0] = self._norm(c01)

        if 'img_gray' in results:
            g = results['img_gray']
            g01 = self._unnorm(g)
            gs = self.gray_strength
            g_gamma = 1.0 + (gamma - 1.0) * gs
            g_gain = 1.0 - (1.0 - gain) * gs
            g_back = None if backlight is None else (1.0 - (1.0 - backlight) * gs)
            g01 = self._degrade(g01, g_gamma, g_gain, self.noise_std * gs, g_back)
            results['img_gray'] = self._norm(g01)

        # Corrupt the (precomputed, otherwise-clean) stereo depth to mimic low-light stereo-matching failure.
        if self.degrade_depth and 'stereo_depth' in results:
            sev = (1.0 - gain)                                  # darker => more corruption
            def _corrupt(d, s):
                scale = float(d[d > 0].abs().mean()) if (d > 0).any() else 1.0
                d = d + torch.randn_like(d) * self.depth_noise_scale * s * scale
                d = d.masked_fill(torch.rand_like(d) < self.depth_dropout * s, 0.0)
                return d.clamp_min(0.0)
            results['stereo_depth'] = _corrupt(results['stereo_depth'], sev)          # color depth: full
            if 'gray_stereo_depth' in results:                                         # gray depth: survives
                results['gray_stereo_depth'] = _corrupt(results['gray_stereo_depth'], sev * self.gray_strength)

        return results
