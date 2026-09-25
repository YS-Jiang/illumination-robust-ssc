# Offline generation of the graded illumination-degradation evaluation sets.
# Applies the same photometric operator as SyntheticIlluminationDegradation (gamma/gain/noise
# in [0,1] space) at fixed strengths. The color pair (image_2/image_3) is degraded at full
# strength; the grayscale pair (image_0/image_1) at the ratio r (default 0.3), reflecting the
# higher SNR of the sensor without a color filter array.
#
# Example (sequence 08, all levels):
#   python preprocess/degrade_semantickitti.py \
#       --src data/semantickitti/sequences/08 --dst data/semantickitti_degraded
import os
import glob
import argparse
import numpy as np
from PIL import Image

LEVELS = {  # level: (gamma, gain); noise_std = 0.04
    'mild': (2.0, 0.6),
    'medium': (3.0, 0.4),
    'strong': (4.0, 0.25),
    'severe': (5.0, 0.2),
    'extreme': (6.5, 0.12),
}


def degrade_dir(src, dst, gamma, gain, noise, seed):
    rng = np.random.RandomState(seed)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, '*.png')))
    for f in files:
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32) / 255.0
        x = np.power(np.clip(im, 1e-6, 1.0), gamma) * gain
        x = np.clip(x + rng.randn(*im.shape).astype(np.float32) * noise, 0.0, 1.0)
        Image.fromarray((x * 255).astype(np.uint8)).save(os.path.join(dst, os.path.basename(f)))
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='sequence dir holding image_0..image_3')
    ap.add_argument('--dst', required=True, help='output root; one subdir per level is created')
    ap.add_argument('--levels', nargs='+', default=list(LEVELS.keys()))
    ap.add_argument('--gray_ratio', type=float, default=0.3,
                    help='fraction of the color degradation applied to the grayscale pair')
    ap.add_argument('--noise', type=float, default=0.04)
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.src))
    for level in args.levels:
        gamma, gain = LEVELS[level]
        r = args.gray_ratio
        g_gamma = 1.0 + (gamma - 1.0) * r
        g_gain = 1.0 - (1.0 - gain) * r
        for cam, (gm, gn, ns) in {
            'image_2': (gamma, gain, args.noise),
            'image_3': (gamma, gain, args.noise),
            'image_0': (g_gamma, g_gain, args.noise * r),
            'image_1': (g_gamma, g_gain, args.noise * r),
        }.items():
            src = os.path.join(args.src, cam)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(args.dst, level, 'sequences', seq, cam)
            n = degrade_dir(src, dst, gm, gn, ns, seed=hash((level, cam)) % (2 ** 31))
            print(f'{level}/{cam}: {n} images -> {dst}', flush=True)


if __name__ == '__main__':
    main()
