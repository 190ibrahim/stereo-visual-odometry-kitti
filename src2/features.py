"""
Feature detection and matching for visual odometry
"""

import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, detector_type='SIFT', distance_threshold=0.35):
        self.detector_type = detector_type.upper()
        self.distance_threshold = distance_threshold
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
    
    def _create_detector(self):
        """Create feature detector"""
        if self.detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=3000)
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher"""
        if self.detector_type == 'SIFT':
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.detector_type == 'ORB':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between two descriptor sets"""
        if desc1 is None or desc2 is None:
            return []
        
        # KNN matching with k=2
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance <= self.distance_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """Draw feature matches for visualization"""
    # Limit number of matches for cleaner visualization
    if len(matches) > max_matches:
        matches = matches[:max_matches]
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches