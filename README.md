# Stereo Visual Odometry (KITTI)

stereo VO pipeline for KITTI odometry benchmark sequences.

## Requirements
- CMake, Make
- OpenCV (>=4.x)
- C++11 toolchain
- Python3 + evo (for evaluation)

## Setup
```bash
git clone https://github.com/190ibrahim/stereo-visual-odometry-kitti.git
cd stereo-visual-odometry-kitti

python3 -m venv .venv && source .venv/bin/activate
pip install evo numpy matplotlib
```

## Dataset (KITTI)
1) Download a sequence from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
2) Place files as:
```
dataset/sequences/04/
  image_0/  # left
  image_1/  # right
  calib.txt
```

## Build
```bash
mkdir -p build && cd build
cmake ..
make -j
```

## Run
```bash
./stereo_visual_odometry
```
