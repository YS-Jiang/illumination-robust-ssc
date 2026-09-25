"""SemanticKITTI-C robustness corruptions.

Applies the ImageNet-C corruption set (via `imagecorruptions`) to the color image at a
given severity. The grayscale second camera receives a lighter corruption, which follows
the physical motivation: no color filter array, so a higher signal-to-noise ratio.
'darkness' is a low-light corruption added here and is not part of ImageNet-C.
Used for evaluation on the SemanticKITTI-C benchmark, for comparability with MonoScene,
VoxFormer and EvSSC.
"""
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SemanticKITTICorruption(object):
    def __init__(self, corruption='shot_noise', severity=3, gray_severity_scale=0.4,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.corruption = corruption
        self.severity = int(severity)
        self.gray_severity_scale = gray_severity_scale   # gray corrupted less (higher SNR)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def _unnorm(self, img):  # (N,3,H,W) normalized -> [0,1]
        return (img * self.std + self.mean).clamp(0.0, 1.0)

    def _norm(self, img01):
        return (img01 - self.mean) / self.std

    def _apply(self, img01, sev):
        """img01: (N,3,H,W) in [0,1]. Apply the named corruption at integer severity sev (1..5)."""
        if sev < 1:
            return img01
        if self.corruption == 'darkness':
            # low-light: multiplicative gain + gamma, 5 severities (custom; ImageNet-C has no darkness)
            gains = [0.6, 0.45, 0.32, 0.22, 0.15][min(sev, 5) - 1]
            gammas = [1.6, 2.2, 3.0, 3.8, 4.6][min(sev, 5) - 1]
            return (img01.clamp_min(1e-6) ** gammas * gains).clamp(0.0, 1.0)
        from imagecorruptions import corrupt
        out = []
        for n in range(img01.shape[0]):
            arr = (img01[n].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # HWC uint8
            c = corrupt(arr, corruption_name=self.corruption, severity=min(max(sev, 1), 5))
            out.append(torch.from_numpy(c.astype(np.float32) / 255.0).permute(2, 0, 1))
        return torch.stack(out, 0).to(img01.dtype)

    def __call__(self, results):
        color = results['img_inputs'][0]
        results['img_inputs'][0] = self._norm(self._apply(self._unnorm(color), self.severity))
        if 'img_gray' in results:
            g_sev = max(int(round(self.severity * self.gray_severity_scale)), 0)
            g = results['img_gray']
            results['img_gray'] = self._norm(self._apply(self._unnorm(g), g_sev))
        return results
