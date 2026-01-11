#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>

// Stereo depth computation functions (like Python utils)
class StereoDepth {
public:
    StereoDepth(const cv::Mat& P0, const cv::Mat& P1);
    cv::Mat computeDepth(const cv::Mat& left_image, const cv::Mat& right_image);
    
private:
    cv::Mat computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image);
    cv::Mat disparityToDepth(const cv::Mat& disparity_map);
    
    cv::Mat P0_, P1_;
    double focal_length_;
    double baseline_;
    cv::Ptr<cv::StereoSGBM> sgbm_matcher_;
};

// Motion estimation using PnP (like Python motion_estimation)
class MotionEstimator {
public:
    MotionEstimator(const cv::Mat& camera_matrix, double max_depth = 3000.0);
    bool estimateMotion(const std::vector<cv::DMatch>& matches,
                       const std::vector<cv::KeyPoint>& keypoints_prev,
                       const std::vector<cv::KeyPoint>& keypoints_current,
                       const cv::Mat& depth_map,
                       cv::Mat& rotation_matrix,
                       cv::Mat& translation_vector);
                       
private:
    cv::Mat camera_matrix_;
    double max_depth_;
    double fx_, fy_, cx_, cy_;
};

// Ground truth loading and evaluation (like Python DataLoader)
class GroundTruthLoader {
public:
    GroundTruthLoader(const std::string& dataset_root, const std::string& sequence);
    bool loadGroundTruth();
    cv::Mat getGroundTruthPose(int frame_idx) const;
    cv::Point3f getGroundTruthPosition(int frame_idx) const;
    
    // Evaluation functions
    double computeTranslationError(const cv::Mat& estimated_pose, int frame_idx) const;
    double computeRotationError(const cv::Mat& estimated_pose, int frame_idx) const;
    void printTrajectoryComparison(const cv::Mat& estimated_pose, int frame_idx) const;
    
    int getNumFrames() const { return ground_truth_poses_.size(); }
    
private:
    std::string dataset_root_;
    std::string sequence_;
    std::vector<cv::Mat> ground_truth_poses_;
};

#endif // POSE_ESTIMATOR_H