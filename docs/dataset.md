## SemanticKITTI

Download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Please follow the command [image2depth_semantickitti](../preprocess/image2depth_semantickitti.sh) to create depth maps and preprocess the annotations for semantic scene completion:

```bash
python tools/preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI
```

### Folder structure

The data is organized in the following format:

```
/semantickittii/
          |-- sequences/
          │       |-- 00/
          │       │   |-- poses.txt
          │       │   |-- calib.txt
          │       │   |-- image_2/
          │       │   |-- image_3/
          │       |   |-- voxels/
          │       |         |- 000000.bin
          │       |         |- 000000.label
          │       |         |- 000000.occluded
          │       |         |- 000000.invalid
          │       |         |- 000005.bin
          │       |         |- 000005.label
          │       |         |- 000005.occluded
          │       |         |- 000005.invalid
          │       |-- 01/
          │       |-- 02/
          │       .
          │       |-- 21/
          |-- labels/
          │       |-- 00/
          │       │   |-- 000000_1_1.npy
          │       │   |-- 000000_1_2.npy
          │       │   |-- 000005_1_1.npy
          │       │   |-- 000005_1_2.npy
          │       |-- 01/
          │       .
          │       |-- 10/
          |-- lidarseg/
          |       |-- 00/
          |       │   |-- labels/
          |       |         ├ 000001.label
          |       |         ├ 000002.label
          |       |-- 01/
          |       |-- 02/
          |       .
          |       |-- 21/
          |-- depth/sequences/
          		  |-- 00/
          		  │   |-- 000000.npy
          		  |   |-- 000001.npy
          		  |-- 01/
                  |-- 02/
                  .
                  |-- 21/
          
```

## Grayscale pair

This work additionally uses the KITTI odometry **grayscale** images (`image_0/`, `image_1/`)
of the same sequences, placed next to `image_2/` and `image_3/`. Grayscale stereo depth is
precomputed with the same MobileStereoNet procedure as the color depth and stored under
`depth_gray/sequences/<seq>/`.
