#include "pose_estimator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

StereoDepth::StereoDepth(const cv::Mat& P0, const cv::Mat& P1) : P0_(P0), P1_(P1) {
    // Extract focal length and baseline (like Python decomposition)
    focal_length_ = P0.at<double>(0, 0);
    baseline_ = -P1.at<double>(0, 3) / P1.at<double>(0, 0);
    
    // Create SGBM matcher (like Python disparity_mapping)
    int num_disparities = 6 * 16;  // 96
    int block_size = 7;
    int num_channels = 1; // Grayscale
    
    sgbm_matcher_ = cv::StereoSGBM::create(
        0,                                           // minDisparity
        num_disparities,                            // numDisparities  
        block_size,                                 // blockSize
        8 * num_channels * block_size * block_size, // P1
        32 * num_channels * block_size * block_size,// P2
        1,                                          // disp12MaxDiff
        63,                                         // preFilterCap
        10,                                         // uniquenessRatio
        150,                                        // speckleWindowSize
        2,                                          // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY             // mode
    );
    
    std::cout << "StereoDepth initialized - focal: " << focal_length_ << ", baseline: " << baseline_ << std::endl;
}

cv::Mat StereoDepth::computeDepth(const cv::Mat& left_image, const cv::Mat& right_image) {
    // 1. Compute disparity (like Python disparity_mapping)
    cv::Mat disparity = computeDisparity(left_image, right_image);
    
    // 2. Convert disparity to depth (like Python depth_mapping)
    cv::Mat depth = disparityToDepth(disparity);
    
    return depth;
}

cv::Mat StereoDepth::computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
    cv::Mat disparity;
    sgbm_matcher_->compute(left_image, right_image, disparity);
    
    // Convert to float and divide by 16 (like Python)
    disparity.convertTo(disparity, CV_32F, 1.0/16.0);
    
    return disparity;
}

cv::Mat StereoDepth::disparityToDepth(const cv::Mat& disparity_map) {
    cv::Mat depth_map = cv::Mat::ones(disparity_map.size(), CV_32F);
    
    // Avoid division by zero (like Python)
    cv::Mat disparity_safe = disparity_map.clone();
    disparity_safe.setTo(0.1, disparity_map <= 0);
    
    // depth = focal_length * baseline / disparity (like Python formula)
    depth_map = (focal_length_ * baseline_) / disparity_safe;
    
    return depth_map;
}

// Motion Estimation Implementation (like Python motion_estimation)
MotionEstimator::MotionEstimator(const cv::Mat& camera_matrix, double max_depth) 
    : camera_matrix_(camera_matrix), max_depth_(max_depth) {
    
    // Extract intrinsic parameters (like Python)
    fx_ = camera_matrix.at<double>(0, 0);
    fy_ = camera_matrix.at<double>(1, 1);
    cx_ = camera_matrix.at<double>(0, 2);
    cy_ = camera_matrix.at<double>(1, 2);
    
    std::cout << "MotionEstimator initialized - fx: " << fx_ << ", fy: " << fy_ 
              << ", cx: " << cx_ << ", cy: " << cy_ << ", max_depth: " << max_depth_ << std::endl;
}

bool MotionEstimator::estimateMotion(const std::vector<cv::DMatch>& matches,
                                   const std::vector<cv::KeyPoint>& keypoints_prev,
                                   const std::vector<cv::KeyPoint>& keypoints_current,
                                   const cv::Mat& depth_map,
                                   cv::Mat& rotation_matrix,
                                   cv::Mat& translation_vector) {
    
    if (matches.empty()) {
        std::cerr << "No matches for motion estimation" << std::endl;
        return false;
    }
    
    // Extract 2D points from matches (like Python)
    std::vector<cv::Point2f> image1_points, image2_points;
    for (const auto& match : matches) {
        image1_points.push_back(keypoints_prev[match.queryIdx].pt);   // Previous frame
        image2_points.push_back(keypoints_current[match.trainIdx].pt); // Current frame
    }
    
    // Convert 2D points from previous frame to 3D using depth (like Python)
    std::vector<cv::Point3f> points_3D;
    std::vector<cv::Point2f> points_2D;
    std::vector<int> outliers;
    
    for (int i = 0; i < image1_points.size(); i++) {
        cv::Point2f pt = image1_points[i];
        int u = static_cast<int>(pt.x);
        int v = static_cast<int>(pt.y);
        
        // Check bounds
        if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
            float z = depth_map.at<float>(v, u);
            
            // Filter out invalid depth (like Python max_depth check)
            if (z > 0 && z < max_depth_) {
                // Convert to 3D (like Python formula: x = z*(u-cx)/fx)
                float x = z * (u - cx_) / fx_;
                float y = z * (v - cy_) / fy_;
                
                points_3D.push_back(cv::Point3f(x, y, z));
                points_2D.push_back(image2_points[i]);  // Corresponding 2D point in current frame
            } else {
                outliers.push_back(i);
            }
        } else {
            outliers.push_back(i);
        }
    }
    
    if (points_3D.size() < 4) {
        std::cerr << "Not enough 3D points for PnP: " << points_3D.size() << std::endl;
        return false;
    }
    
    // Apply PnP RANSAC (like Python cv2.solvePnPRansac)
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(points_3D, points_2D, camera_matrix_, cv::noArray(),
                                     rvec, tvec, false, 100, 8.0, 0.99, inliers);
    
    if (!success) {
        std::cerr << "PnP RANSAC failed" << std::endl;
        return false;
    }
    
    // Convert rotation vector to matrix (like Python cv2.Rodrigues)
    cv::Rodrigues(rvec, rotation_matrix);
    translation_vector = tvec;
    
    std::cout << "Motion estimated from " << points_3D.size() << " 3D points, "
              << inliers.size() << " inliers" << std::endl;
    
    return true;
}

// Ground Truth Implementation (like Python DataLoader)
GroundTruthLoader::GroundTruthLoader(const std::string& dataset_root, const std::string& sequence) 
    : dataset_root_(dataset_root), sequence_(sequence) {
}

bool GroundTruthLoader::loadGroundTruth() {
    std::string poses_file = dataset_root_ + "/poses/" + sequence_ + ".txt";
    std::ifstream file(poses_file);
    
    if (!file.is_open()) {
        std::cerr << "Could not open ground truth file: " << poses_file << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        
        // Read 12 values from each line (3x4 transformation matrix)
        while (iss >> val) {
            values.push_back(val);
        }
        
        if (values.size() == 12) {
            // Create 4x4 homogeneous matrix (like Python)
            cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
            
            // Fill 3x4 part
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    pose.at<double>(i, j) = values[i * 4 + j];
                }
            }
            
            ground_truth_poses_.push_back(pose);
        }
    }
    
    file.close();
    std::cout << "Loaded " << ground_truth_poses_.size() << " ground truth poses" << std::endl;
    return !ground_truth_poses_.empty();
}

cv::Mat GroundTruthLoader::getGroundTruthPose(int frame_idx) const {
    if (frame_idx >= 0 && frame_idx < ground_truth_poses_.size()) {
        return ground_truth_poses_[frame_idx];
    }
    return cv::Mat::eye(4, 4, CV_64F);
}

cv::Point3f GroundTruthLoader::getGroundTruthPosition(int frame_idx) const {
    if (frame_idx >= 0 && frame_idx < ground_truth_poses_.size()) {
        cv::Mat pose = ground_truth_poses_[frame_idx];
        return cv::Point3f(
            pose.at<double>(0, 3),  // x
            pose.at<double>(1, 3),  // y
            pose.at<double>(2, 3)   // z
        );
    }
    return cv::Point3f(0, 0, 0);
}

double GroundTruthLoader::computeTranslationError(const cv::Mat& estimated_pose, int frame_idx) const {
    if (frame_idx >= ground_truth_poses_.size()) return -1.0;
    
    cv::Mat gt_pose = ground_truth_poses_[frame_idx];
    
    // Extract translation vectors
    cv::Point3f gt_pos(gt_pose.at<double>(0, 3), gt_pose.at<double>(1, 3), gt_pose.at<double>(2, 3));
    cv::Point3f est_pos(estimated_pose.at<double>(0, 3), estimated_pose.at<double>(1, 3), estimated_pose.at<double>(2, 3));
    
    // Euclidean distance
    cv::Point3f diff = gt_pos - est_pos;
    return sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}

double GroundTruthLoader::computeRotationError(const cv::Mat& estimated_pose, int frame_idx) const {
    if (frame_idx >= ground_truth_poses_.size()) return -1.0;
    
    cv::Mat gt_pose = ground_truth_poses_[frame_idx];
    
    // Extract rotation matrices
    cv::Mat gt_R = gt_pose(cv::Rect(0, 0, 3, 3));
    cv::Mat est_R = estimated_pose(cv::Rect(0, 0, 3, 3));
    
    // Compute rotation error using trace formula
    cv::Mat R_diff = gt_R.t() * est_R;
    double trace = cv::trace(R_diff)[0];
    double cos_angle = (trace - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // Clamp to [-1,1]
    
    return acos(cos_angle) * 180.0 / CV_PI; // Convert to degrees
}

void GroundTruthLoader::printTrajectoryComparison(const cv::Mat& estimated_pose, int frame_idx) const {
    if (frame_idx >= ground_truth_poses_.size()) return;
    
    cv::Point3f gt_pos = getGroundTruthPosition(frame_idx);
    cv::Point3f est_pos(estimated_pose.at<double>(0, 3), estimated_pose.at<double>(1, 3), estimated_pose.at<double>(2, 3));
    
    double trans_error = computeTranslationError(estimated_pose, frame_idx);
    double rot_error = computeRotationError(estimated_pose, frame_idx);
    
    std::cout << "Frame " << frame_idx << ":";
    std::cout << " GT=" << std::fixed << std::setprecision(2) << "[" << gt_pos.x << ", " << gt_pos.y << ", " << gt_pos.z << "]";
    std::cout << " EST=" << "[" << est_pos.x << ", " << est_pos.y << ", " << est_pos.z << "]";
    std::cout << " TransErr=" << std::setprecision(3) << trans_error << "m";
    std::cout << " RotErr=" << std::setprecision(2) << rot_error << "°" << std::endl;
}