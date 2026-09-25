# MobileStereoNet weights

The pretrained MobileStereoNet weights are not included in this repository, because they
belong to the MobileStereoNet authors.

Download `MSNet3D_SF_DS_KITTI2015.ckpt` from the official repository,

    https://github.com/cogsys-tuebingen/mobilestereonet

and place the file in this directory. `../image2depth_semantickitti.sh` expects it at
`preprocess/mobilestereonet/MSNet3D_SF_DS_KITTI2015.ckpt`.

`filenames/` lists the color stereo pairs and `filenames_gray/` the grayscale stereo pairs
of KITTI odometry sequences 00 to 21. The lists for KITTI-360 and for the keyframe and test
splits that ship with upstream MobileStereoNet are not used by this work and are not kept
here.
