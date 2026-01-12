import numpy as np
import matplotlib.pyplot as plt
from calibration import CameraCalibration
from features import FeatureDetector
from stereo_depth import StereoDepth
from motion_estimation import MotionEstimator
import cv2
import os
import yaml

class StereoVisualOdometry:
    def __init__(self, sequence='00'):
        self.sequence = sequence
        self.sequence_dir = f'../dataset/sequences/{sequence}/'
        
        # Load configuration
        with open('../config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.calibration = CameraCalibration('../dataset', sequence)
        self.feature_detector = FeatureDetector(
            detector_type=self.config.get('vo', {}).get('detector', 'SIFT'),
            distance_threshold=self.config.get('vo', {}).get('distance_threshold', 0.35)
        )
        self.stereo_depth = StereoDepth(
            self.calibration.fx,
            self.calibration.baseline
        )
        self.motion_estimator = MotionEstimator(
            self.calibration.K,
            max_depth=self.config.get('vo', {}).get('max_depth', 3000)
        )
        
        # Load images
        self.left_images = sorted(os.listdir(self.sequence_dir + 'image_0/'))
        self.right_images = sorted(os.listdir(self.sequence_dir + 'image_1/'))
        self.num_frames = len(self.left_images)
        
        # Load ground truth if available
        try:
            self.ground_truth = self._load_ground_truth()
        except:
            self.ground_truth = None
            print("Warning: Could not load ground truth")
        
        print(f"Loaded {self.num_frames} image pairs for sequence {sequence}")
        
    def _load_ground_truth(self):
        """Load ground truth poses from KITTI format"""
        poses_file = f'../dataset/poses/{self.sequence}.txt'
        poses_data = np.loadtxt(poses_file)
        
        poses = []
        for pose_data in poses_data:
            T = np.eye(4)
            T[:3, :] = pose_data.reshape(3, 4)
            poses.append(T)
        return np.array(poses)
    
    def run(self, max_frames=None, plot=True):
        """Run visual odometry with visualization"""
        
        if max_frames is None:
            num_frames = self.num_frames
        else:
            num_frames = min(max_frames, self.num_frames)
            
        subset = self.config.get('subset')
        if subset is not None:
            num_frames = min(subset, num_frames)
        
        print(f"Processing {num_frames} frames with {self.config.get('vo', {}).get('detector', 'SIFT')} detector")
        
        # Initialize visualization
        if plot:
            plt.ion()  # Turn on interactive mode
            fig = plt.figure(figsize=(14, 14))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=-20, azim=270)
            
            # Plot ground truth if available
            if self.ground_truth is not None:
                xs = self.ground_truth[:num_frames, 0, 3]
                ys = self.ground_truth[:num_frames, 1, 3]
                zs = self.ground_truth[:num_frames, 2, 3]
                
                # Plotting range for better visualisation
                ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
                ax.plot(xs, ys, zs, c='dimgray')
            
            ax.set_title("Ground Truth vs Estimated Trajectory")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            plt.show(block=False)
            plt.draw()
        
        # Initialize trajectory
        trajectory = np.zeros((num_frames, 4, 4))
        trajectory[0] = np.eye(4)
        current_pose = np.eye(4)
        
        # Process frames
        for i in range(num_frames - 1):
            # Load images
            img_left = cv2.imread(self.sequence_dir + 'image_0/' + self.left_images[i], 0)
            img_right = cv2.imread(self.sequence_dir + 'image_1/' + self.right_images[i], 0)
            img_next = cv2.imread(self.sequence_dir + 'image_0/' + self.left_images[i+1], 0)
            
            # Compute stereo depth
            depth_map = self.stereo_depth.compute_depth(img_left, img_right)
            
            # Extract features
            kp1, desc1 = self.feature_detector.detect_and_compute(img_left)
            kp2, desc2 = self.feature_detector.detect_and_compute(img_next)
            
            # Match features
            matches = self.feature_detector.match_features(desc1, desc2)
            
            if len(matches) < 50:
                print(f"Warning: Only {len(matches)} matches at frame {i}")
                trajectory[i+1] = current_pose
                continue
                
            # Estimate motion using PnP RANSAC
            try:
                T, inliers = self._estimate_motion(matches, kp1, kp2, depth_map)
                
                # Update pose: pose = pose @ inv(T)
                current_pose = current_pose @ np.linalg.inv(T)
                trajectory[i+1] = current_pose
                
                if i % 10 == 0:
                    pos = current_pose[:3, 3]
                    print(f'{i} frames computed - Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] - Matches: {len(matches)}, Inliers: {inliers}')
                
                # Update plot
                if plot:
                    xs = trajectory[:i+2, 0, 3]
                    ys = trajectory[:i+2, 1, 3] 
                    zs = trajectory[:i+2, 2, 3]
                    plt.plot(xs, ys, zs, c='darkorange')
                    plt.pause(1e-32)
                    
            except Exception as e:
                print(f"Motion estimation failed at frame {i}: {e}")
                trajectory[i+1] = trajectory[i]  # Keep previous pose
        
        if plot:
            plt.show()
        
        print(f'All frames computed - Final position: {current_pose[:3, 3]}')
        
        # Save trajectory in KITTI format
        output_file = f'../results/trajectory_{self.sequence}.txt'
        self._save_trajectory(trajectory[:num_frames], output_file)
        
        return trajectory[:num_frames]
    
    def _estimate_motion(self, matches, kp1, kp2, depth_map):
        """Motion estimation using PnP RANSAC"""
        # Extract 2D points from matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Build 3D points using depth map
        points_3D = []
        valid_pts2 = []
        
        for (u, v), pt2 in zip(pts1, pts2):
            z = depth_map[int(v), int(u)]
            
            if z > self.config.get('vo', {}).get('max_depth', 3000) or z <= 0:
                continue
                
            # Convert to 3D using camera intrinsics
            x = z * (u - self.calibration.cx) / self.calibration.fx
            y = z * (v - self.calibration.cy) / self.calibration.fy
            
            points_3D.append([x, y, z])
            valid_pts2.append(pt2)
        
        if len(points_3D) < 10:
            raise ValueError("Not enough 3D points")
        
        points_3D = np.array(points_3D, dtype=np.float32)
        valid_pts2 = np.array(valid_pts2, dtype=np.float32)
        
        # Camera intrinsic matrix
        K = np.array([[self.calibration.fx, 0, self.calibration.cx],
                      [0, self.calibration.fy, self.calibration.cy],
                      [0, 0, 1]], dtype=np.float32)
        
        # PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3D, valid_pts2, K, None)
        
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
        print(f"Trajectory saved to: {filename}")


if __name__ == "__main__":
    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    sequence = config.get('dataset', {}).get('sequence', '00')
    max_frames = config.get('vo', {}).get('max_frames', None)
    plot_enabled = config.get('vo', {}).get('plot', True)
    
    print(f"Starting Visual Odometry - Sequence: {sequence}")
    if max_frames:
        print(f"Processing max {max_frames} frames")
    
    # Create and run visual odometry
    vo = StereoVisualOdometry(sequence=sequence)
    trajectory = vo.run(max_frames=max_frames, plot=plot_enabled)