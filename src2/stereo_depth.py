import cv2
import numpy as np

class StereoDepth:
    def __init__(self, focal_length, baseline):
        self.focal_length = focal_length
        self.baseline = baseline
        
        # SGBM (semi-global block matching) algorithm
        # finds corresponding pixels between left and right images
        # better than simple block matching, handles textureless areas
        self.matcher = cv2.StereoSGBM_create(
            # max disparity range in pixels (must be divisible by 16)
            # 96 works well for KITTI, captures depth up to ~40 meters
            numDisparities=96,
            # minimum disparity to search (usually 0)
            minDisparity=0,
            # size of matching window (odd number)
            # 7x7 is good balance between speed and accuracy
            blockSize=7,
            # P1: penalty for small disparity changes (smoothness)
            # smaller changes are ok, so small penalty
            P1=8 * 1 * 7 ** 2,
            # P2: penalty for large disparity changes
            # big jumps should be rare, so larger penalty
            # this makes the depth map smoother
            P2=32 * 1 * 7 ** 2,
            # 3-way mode is most accurate but slower
            # checks left-right and up-down consistency
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def compute_depth(self, left_img, right_img):
        """Compute depth map using stereo matching"""
        # find how much each pixel shifted between left and right images
        # opencv returns disparity * 16 for precision, so divide by 16
        disparity = self.matcher.compute(left_img, right_img).astype(np.float32) / 16.0
        
        # replace zeros and invalid values to avoid division errors
        # -1 means no match found, 0 would give infinite depth
        disparity[disparity == 0.0] = 0.1
        disparity[disparity == -1.0] = 0.1
        
        # calculate actual depth using stereo geometry
        # larger disparity = closer object, smaller disparity = farther
        # this is the key formula for stereo vision
        depth = np.ones(disparity.shape)
        depth = (self.focal_length * self.baseline) / disparity
        
        return depth, disparity
    
    def visualize_depth_and_disparity(self, depth, disparity, frame_idx=0):
        """Visualize depth and disparity maps"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Disparity map
        im1 = ax1.imshow(disparity, cmap='plasma', vmin=0, vmax=64)
        ax1.set_title(f'Disparity Map - Frame {frame_idx}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Disparity (pixels)')
        
        # Depth map (clip extreme values for better visualization)
        depth_clipped = np.clip(depth, 0, 100)
        im2 = ax2.imshow(depth_clipped, cmap='viridis')
        ax2.set_title(f'Depth Map - Frame {frame_idx}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Depth (m)')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)