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
        
        # get P0 (left camera projection matrix)
        # this is a 3x4 matrix that maps 3d points to 2d image coordinates
        P0_values = [float(x) for x in lines[0].strip().split()[1:]]
        self.P0 = np.array(P0_values).reshape(3, 4)
        
        # get P1 (right camera projection matrix)
        # same as P0 but for the right camera
        P1_values = [float(x) for x in lines[1].strip().split()[1:]]
        self.P1 = np.array(P1_values).reshape(3, 4)
        
        # extract focal lengths and principal point from P0
        # fx/fy: how many pixels per unit distance (zoom factor)
        # cx/cy: center of the image where optical axis hits
        self.fx = self.P0[0, 0]  
        self.fy = self.P0[1, 1]  
        self.cx = self.P0[0, 2]  
        self.cy = self.P0[1, 2]  
        
        # build K matrix (intrinsic camera parameters)
        # we need this for converting between 2d and 3d points
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]])
        
        # baseline: physical distance between left and right cameras in meters
        # we need this to calculate depth from disparity
        # formula: depth = (fx * baseline) / disparity
        self.baseline = abs(self.P1[0, 3] / self.P1[0, 0])

