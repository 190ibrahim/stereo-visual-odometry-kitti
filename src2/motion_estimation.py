import cv2
import numpy as np

class MotionEstimator:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.max_depth = 3000  # Good default for KITTI
        self.ransac_iterations = 1000
        self.reprojection_error = 8.0
        self.confidence = 0.99
        
        # Extract intrinsic parameters
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1] 
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def estimate_motion(self, matches, keypoints_prev, keypoints_curr, depth_map):
        """Estimate camera motion using PnP RANSAC"""
        if len(matches) < 8:
            return None, None, 0
        
        # Extract matching points
        points_2d_prev = []
        points_2d_curr = []
        
        for match in matches:
            kp_prev = keypoints_prev[match.queryIdx]
            kp_curr = keypoints_curr[match.trainIdx]
            
            points_2d_prev.append(kp_prev.pt)
            points_2d_curr.append(kp_curr.pt)
        
        points_2d_prev = np.array(points_2d_prev, dtype=np.float32)
        points_2d_curr = np.array(points_2d_curr, dtype=np.float32)
        
        # Build 3D points from previous frame using depth map
        points_3d = []
        valid_points_2d = []
        
        for i, (u, v) in enumerate(points_2d_prev):
            u_int, v_int = int(u), int(v)
            
            # Check bounds and depth
            if (0 <= u_int < depth_map.shape[1] and 
                0 <= v_int < depth_map.shape[0]):
                
                depth = depth_map[v_int, u_int]
                
                if 0 < depth < self.max_depth:
                    # Convert to 3D coordinates 
                    x = depth * (u - self.cx) / self.fx
                    y = depth * (v - self.cy) / self.fy
                    z = depth
                    
                    points_3d.append([x, y, z])
                    valid_points_2d.append(points_2d_curr[i])
        
        if len(points_3d) < 8:
            return None, None, 0
        
        points_3d = np.array(points_3d, dtype=np.float32)
        valid_points_2d = np.array(valid_points_2d, dtype=np.float32)
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, valid_points_2d, self.camera_matrix, None,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_error,
            confidence=self.confidence
        )
        
        if not success:
            return None, None, 0
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        num_inliers = len(inliers)
        
        return R, tvec, num_inliers