#include "feature_detector.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>  // For SIFT

StereoFeatureDetector::StereoFeatureDetector(const std::string& detector_type) 
    : detector_type_(detector_type) {
    
    if (detector_type_ == "ORB") {
        detector_ = cv::ORB::create();  // Default ~500 features
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);  // crossCheck=false for knnMatch
        std::cout << "Using ORB detector with default features" << std::endl;
    } else if (detector_type_ == "SIFT") {
        try {
            detector_ = cv::xfeatures2d::SIFT::create(); // Default unlimited features
            matcher_ = cv::BFMatcher::create(cv::NORM_L2, false);  // crossCheck=false for knnMatch
            std::cout << "Using SIFT detector with default features" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "SIFT not available (need opencv_contrib). Using ORB instead." << std::endl;
            detector_ = cv::ORB::create();
            matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
            detector_type_ = "ORB";
        }
    } else {
        std::cerr << "Unknown detector type: " << detector_type_ << ". Using ORB." << std::endl;
        detector_ = cv::ORB::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
        detector_type_ = "ORB";
    }
}

void StereoFeatureDetector::detectFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (image.empty()) {
        std::cerr << "Empty image passed to detectFeatures" << std::endl;
        return;
    }
    
    detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    std::cout << "Detected " << keypoints.size() << " keypoints" << std::endl;
}

void StereoFeatureDetector::matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches) {
    if (desc1.empty() || desc2.empty()) {
        std::cerr << "Empty descriptors passed to matchFeatures" << std::endl;
        return;
    }
    
    // Use knnMatch with k=2 like Python code
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(desc1, desc2, knn_matches, 2);
    
    // Apply Lowe's ratio test (like Python: distance_threshold = 0.35 from config)
    const float distance_threshold = 0.35f; // From your config.yaml
    matches.clear();
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() == 2) {
            const cv::DMatch& match1 = match_pair[0]; // Best match
            const cv::DMatch& match2 = match_pair[1]; // Second best match
            
            // Lowe's ratio test: match1.distance <= distance_threshold * match2.distance
            if (match1.distance <= distance_threshold * match2.distance) {
                matches.push_back(match1);
            }
        }
    }
    
    std::cout << "Found " << knn_matches.size() << " raw matches, filtered to " << matches.size() << " good matches" << std::endl;
}