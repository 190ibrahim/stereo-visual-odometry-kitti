#pragma once
#include <opencv2/opencv.hpp>
#include <string>

void detectFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                    const std::string& detector_type, int n_features);

void matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);

// Filter stereo matches by epipolar constraint and positive disparity
void filterStereoMatches(
    const std::vector<cv::DMatch>& stereo_matches,
    const std::vector<cv::KeyPoint>& keypoints_left,
    const std::vector<cv::KeyPoint>& keypoints_right,
    std::vector<cv::Point2f>& matches0,
    std::vector<cv::Point2f>& matches1,
    std::vector<cv::Point2f>& fails);