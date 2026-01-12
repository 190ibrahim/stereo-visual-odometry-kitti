# Stereo Visual Odometry (KITTI)

stereo VO pipeline for KITTI odometry benchmark sequences.

## Setup
```bash
git clone https://github.com/190ibrahim/stereo-visual-odometry-kitti.git
cd stereo-visual-odometry-kitti

python3 -m venv .venv && source .venv/bin/activate
pip install evo numpy matplotlib
```

## Dataset (KITTI)
1) From the KITTI website http://www.cvlibs.net/datasets/kitti/eval_odometry.php , download:

* **Odometry dataset (grayscale images)**
* **Odometry ground truth poses**

2) Place files as:

```
dataset/
├── sequences/00/
│   ├── image_0/    # left grayscale images
│   ├── image_1/    # right grayscale images
│   ├── calib.txt   # stereo calibration
│   └── times.txt   
└── poses/
    └── 00.txt       # ground truth poses
```

## Build
```bash
mkdir -p build && cd build
cmake ..
make 
```

## Run
```bash
./stereo_visual_odometry
```
