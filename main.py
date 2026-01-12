import numpy as np
import matplotlib.pyplot as plt
from src2.calibration import CameraCalibration
from src2.features import FeatureDetector, visualize_matches
from src2.stereo_depth import StereoDepth
from src2.motion_estimation import MotionEstimator
from src2.visualization import TrajectoryVisualizer
import cv2
import os
import yaml

class StereoVisualOdometry:
    def __init__(self, sequence='00'):
        self.sequence = sequence
        self.sequence_dir = f'dataset/sequences/{sequence}/'
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.calibration = CameraCalibration('dataset', sequence)
        self.feature_detector = FeatureDetector(
            detector_type=self.config.get('vo', {}).get('detector', 'sift')
        )
        self.stereo_depth = StereoDepth(
            self.calibration.fx,
            self.calibration.baseline
        )
        self.motion_estimator = MotionEstimator(self.calibration.K)
        
        # Load images
        self.left_images = sorted(os.listdir(self.sequence_dir + 'image_0/'))
        self.right_images = sorted(os.listdir(self.sequence_dir + 'image_1/'))
        self.num_frames = len(self.left_images)
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        
    def _load_ground_truth(self):
        """Load ground truth poses from KITTI format"""
        poses_file = f'dataset/poses/{self.sequence}.txt'
        poses_data = np.loadtxt(poses_file)
        
        poses = []
        for pose_data in poses_data:
            T = np.eye(4)
            T[:3, :] = pose_data.reshape(3, 4)
            poses.append(T)
        return np.array(poses)
    
    def run(self, max_frames=None, plot=True):
        """Run visual odometry with visualization"""
        
        # limit how many frames to process
        # useful for testing without running the whole sequence
        num_frames = max_frames if max_frames else self.num_frames
        num_frames = min(num_frames, self.num_frames)
        
        print(f"Processing {num_frames} frames (max_frames={max_frames}, total available={self.num_frames})")
        
        # set up 3d plot for trajectory
        if plot:
            visualizer = TrajectoryVisualizer(self.ground_truth)
            visualizer.setup_plot()
        
        # trajectory stores camera position at each frame
        # 4x4 transformation matrix for each frame (rotation + translation)
        trajectory = np.zeros((num_frames, 4, 4))
        trajectory[0] = np.eye(4)  # start at origin
        current_pose = np.eye(4)  # identity matrix = no movement yet
        
        # Process frames
        for i in range(num_frames - 1):
            # Load images
            img_left = cv2.imread(self.sequence_dir + 'image_0/' + self.left_images[i], 0)
            img_right = cv2.imread(self.sequence_dir + 'image_1/' + self.right_images[i], 0)
            img_next = cv2.imread(self.sequence_dir + 'image_0/' + self.left_images[i+1], 0)
            
            # Compute stereo depth
            depth_map, disparity = self.stereo_depth.compute_depth(img_left, img_right)
            
            # Extract features
            kp1, desc1 = self.feature_detector.detect_and_compute(img_left)
            kp2, desc2 = self.feature_detector.detect_and_compute(img_next)
            
            # find which features in current frame match features in next frame
            # uses ratio test to filter bad matches
            matches = self.feature_detector.match_features(desc1, desc2)
            
            # show live video of what camera sees
            # helps you see what's being processed
            viz_config = self.config.get('visualization', {})
            if viz_config.get('show_video_stream', True):
                cv2.imshow('Stereo Visual Odometry - Left Camera', img_left)
                cv2.waitKey(1)  # needed to update window, 1ms delay
            
            # show detailed visualizations every N frames
            # showing every frame would be too slow
            update_interval = viz_config.get('update_interval', 100)
            if update_interval > 0 and i % update_interval == 0:
                
                # depth map shows how far each pixel is
                # disparity map shows pixel shift between left/right images
                if (viz_config.get('show_depth_map', False) or viz_config.get('show_disparity_map', False)):
                    self.stereo_depth.visualize_depth_and_disparity(depth_map, disparity, i)
                
                # draw lines connecting matched features between frames
                # good for debugging feature tracking
                if viz_config.get('show_matches', False):
                    visualize_matches(img_left, kp1, img_next, kp2, matches, i)
            
            # skip this frame if not enough good matches
            # 50 is minimum for reliable motion estimation
            # just keep previous position if we can't estimate motion
            if len(matches) < 50:
                trajectory[i+1] = current_pose
                continue
                
            # estimate how camera moved between frames
            # PnP RANSAC: finds transformation from 3d points to 2d image points
            # RANSAC filters out bad matches (outliers)
            try:
                T, inliers = self._estimate_motion(matches, kp1, kp2, depth_map)
                
                # update camera position
                # T is movement from current to next frame
                # we invert it to get absolute position
                current_pose = current_pose @ np.linalg.inv(T)
                trajectory[i+1] = current_pose
                
                if i % 50 == 0:
                    print(f'Frame {i}')
                
                # Update plot
                visualizer.update_trajectory(trajectory, i+1)
                    
            except:
                trajectory[i+1] = trajectory[i]
        
        # Clean up video stream window
        if self.config.get('visualization', {}).get('show_video_stream', True):
            cv2.destroyAllWindows()
        
        if plot:
            visualizer.show_final_plot()
        
        print('Processing complete')
        
        # Save trajectory in KITTI format
        output_file = f'results/trajectory_{self.sequence}.txt'
        self._save_trajectory(trajectory[:num_frames], output_file)
        
        return trajectory[:num_frames]
    
    def _estimate_motion(self, matches, kp1, kp2, depth_map):
        """Motion estimation using PnP RANSAC"""
        # get pixel locations of matched features
        # pts1: where features are in current frame
        # pts2: where same features are in next frame
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # convert 2d points to 3d using depth map
        # we need 3d points in current frame to estimate motion
        points_3D = []
        valid_pts2 = []
        
        for (u, v), pt2 in zip(pts1, pts2):
            z = depth_map[int(v), int(u)]  # get depth at this pixel
            
            # filter out bad depth values
            # >3000 is too far (probably error), <=0 is invalid
            if z > 3000 or z <= 0:
                continue
                
            # convert pixel (u,v) to 3d point (x,y,z)
            # this is reverse of camera projection
            # subtract principal point then scale by depth/focal length
            x = z * (u - self.calibration.cx) / self.calibration.fx
            y = z * (v - self.calibration.cy) / self.calibration.fy
            
            points_3D.append([x, y, z])
            valid_pts2.append(pt2)
        
        # need at least 10 points for reliable estimate
        # PnP needs minimum 4, but more is better
        if len(points_3D) < 10:
            raise ValueError("Not enough 3D points")
        
        points_3D = np.array(points_3D, dtype=np.float32)
        valid_pts2 = np.array(valid_pts2, dtype=np.float32)
        
        # PnP RANSAC: perspective-n-point with random sample consensus
        # finds camera rotation and translation that best explains the point matches
        # RANSAC tries many random subsets and picks the best one
        # this removes outliers (wrong matches) automatically
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3D, valid_pts2, self.calibration.K, None)
        
        if not success or inliers is None:
            raise ValueError("PnP RANSAC failed")
        
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        
        return T, len(inliers)
    
    def _save_trajectory(self, trajectory, filename):
        """Save trajectory in KITTI format"""
        with open(filename, 'w') as f:
            for pose in trajectory:
                pose_line = pose[:3, :].flatten()
                f.write(' '.join(map(str, pose_line)) + '\n')


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    sequence = config.get('dataset', {}).get('sequence', '00')
    max_frames = config.get('vo', {}).get('max_frames', None)
    plot_enabled = config.get('vo', {}).get('plot', True)
    
    print(f"Starting Visual Odometry - Sequence: {sequence}")
    
    # Create and run visual odometry
    vo = StereoVisualOdometry(sequence=sequence)
    trajectory = vo.run(max_frames=max_frames, plot=plot_enabled)