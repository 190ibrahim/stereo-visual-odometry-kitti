"""
Camera calibration data loader for KITTI dataset
"""

import numpy as np
import os

class CameraCalibration:
    def __init__(self, dataset_root, sequence):
        self.dataset_root = dataset_root
        self.sequence = sequence
        self.load_calibration()
    
    def load_calibration(self):
        """Load camera calibration parameters from KITTI calib.txt"""
        calib_file = os.path.join(self.dataset_root, 'sequences', self.sequence, 'calib.txt')
        
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # Parse P0 (left camera projection matrix)
        P0_values = [float(x) for x in lines[0].strip().split()[1:]]
        self.P0 = np.array(P0_values).reshape(3, 4)
        
        # Parse P1 (right camera projection matrix) 
        P1_values = [float(x) for x in lines[1].strip().split()[1:]]
        self.P1 = np.array(P1_values).reshape(3, 4)
        
        # Extract intrinsic parameters from P0
        self.fx = self.P0[0, 0]  # focal length x
        self.fy = self.P0[1, 1]  # focal length y
        self.cx = self.P0[0, 2]  # principal point x
        self.cy = self.P0[1, 2]  # principal point y
        
        # Camera matrix
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]])
        
        # Baseline (distance between cameras)
        self.baseline = abs(self.P1[0, 3] / self.P1[0, 0])
        
        print(f"Camera calibration loaded:")
        print(f"  Focal length: {self.fx:.3f}")
        print(f"  Principal point: ({self.cx:.3f}, {self.cy:.3f})")
        print(f"  Baseline: {self.baseline:.6f}")

def load_ground_truth(dataset_root, sequence):
    """Load ground truth poses"""
    poses_file = os.path.join(dataset_root, 'poses', f'{sequence}.txt')
    
    if not os.path.exists(poses_file):
        print(f"Warning: Ground truth file {poses_file} not found!")
        return None
    
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            # Convert 12-element vector to 4x4 homogeneous matrix
            pose = np.eye(4)
            pose[0, :] = values[0:4]
            pose[1, :] = values[4:8] 
            pose[2, :] = values[8:12]
            poses.append(pose)
    
    return np.array(poses)