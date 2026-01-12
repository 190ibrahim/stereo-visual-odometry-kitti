import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, detector_type='SIFT'):
        self.detector_type = detector_type.upper()
        # lowe's ratio test threshold
        # 0.35 is more strict than the usual 0.7-0.8, reduces false matches
        # lower value = fewer but better quality matches
        self.distance_threshold = 0.35
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
    
    def _create_detector(self):
        """Create feature detector"""
        if self.detector_type == 'SIFT':
            # SIFT: finds corners and blobs, works well for most scenes
            # slower but more accurate than ORB
            return cv2.SIFT_create()
        elif self.detector_type == 'ORB':
            # ORB: faster than SIFT but less accurate
            # 3000 features is enough for good tracking
            return cv2.ORB_create(nfeatures=3000)
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher"""
        if self.detector_type == 'SIFT':
            # SIFT uses floating point descriptors, so we use L2 distance
            # crossCheck=False because we're doing ratio test instead
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.detector_type == 'ORB':
            # ORB uses binary descriptors, so we use hamming distance
            # hamming counts bit differences, faster for binary data
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        # finds interesting points (corners, edges) and describes them
        # descriptors are like fingerprints for each keypoint
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between two descriptor sets"""
        # k=2 means find the 2 best matches for each feature
        # we need 2 to do the ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # lowe's ratio test: filters out bad matches
        # idea: good match should be much better than second best match
        # if best and second best are similar, probably wrong match
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair  # m is best, n is second best
                # only keep if best is clearly better than second best
                if m.distance <= self.distance_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches

def visualize_matches(img1, kp1, img2, kp2, matches, frame_idx=0):
    """Visualize feature matches between two images with connecting lines"""
    import matplotlib.pyplot as plt
    
    # Create match visualization using OpenCV
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0),  # Green for matches
        singlePointColor=(255, 0, 0),  # Red for unmatched points
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Display with matplotlib
    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches, cmap='gray')
    plt.title(f'Feature Matches - Frame {frame_idx} to {frame_idx+1} ({len(matches)} matches)')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.5)

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """Draw feature matches for visualization"""
    # Limit number of matches for cleaner visualization
    matches = matches[:max_matches]
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches