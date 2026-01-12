import cv2
import numpy as np
import yaml

class StereoDepth:
    def __init__(self, focal_length, baseline):
        self.focal_length = focal_length
        self.baseline = baseline
        
        # Load configuration
        with open('../config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # SGBM parameters from config
        sgbm_config = config.get('sgbm', {})
        self.num_disparities = sgbm_config.get('num_disparities', 96)  # Default 6*16
        self.block_size = sgbm_config.get('block_size', 7)             # Default block size
        
        # Initialize SGBM matcher
        self.matcher = cv2.StereoSGBM_create(
            numDisparities=self.num_disparities,
            minDisparity=0,
            blockSize=self.block_size,
            P1=8 * 1 * self.block_size ** 2,    # 8 * num_channels * block_size^2
            P2=32 * 1 * self.block_size ** 2,   # 32 * num_channels * block_size^2 
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        print(f"StereoDepth initialized - focal: {focal_length}, baseline: {baseline}")
    
    def compute_depth(self, left_img, right_img):
        """Compute depth map using stereo matching"""
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Compute disparity
        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Avoid division by zero
        disparity[disparity == 0.0] = 0.1
        disparity[disparity == -1.0] = 0.1
        
        # Compute depth: depth = (focal_length * baseline) / disparity
        depth = np.ones(disparity.shape)
        depth = (self.focal_length * self.baseline) / disparity
        
        return depth
    
    def visualize_disparity(self, disparity):
        """Visualize disparity map"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.imshow(disparity, cmap='plasma')
        plt.colorbar(label='Disparity (pixels)')
        plt.title('Disparity Map')
        plt.show()