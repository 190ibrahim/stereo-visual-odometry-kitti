# Stereo Visual Odometry (KITTI)

A Python implementation of stereo visual odometry for the KITTI odometry benchmark sequences.

## Installation
```bash
git clone https://github.com/190ibrahim/stereo-visual-odometry-kitti.git
cd stereo-visual-odometry-kitti
git checkout python-stereo-vo
# Install dependencies
pip install opencv-python numpy matplotlib pyyaml
```

## Dataset Setup (KITTI)
1) Download from KITTI website: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
   - **Odometry dataset (grayscale images)**
   - **Odometry ground truth poses**

2) Organize files as:
```
dataset/
├── sequences/
│   └── 00/
│       ├── image_0/    # left grayscale images
│       ├── image_1/    # right grayscale images
│       ├── calib.txt   # stereo calibration
│       └── times.txt   
└── poses/
    └── 00.txt          # ground truth poses
```

## Usage
```bash
python main.py
```