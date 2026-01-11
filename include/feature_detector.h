#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

class StereoFeatureDetector {
public:
    StereoFeatureDetector(const std::string& detector_type);
    void detectFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches);
    
private:
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::string detector_type_;
};

#endif // FEATURE_DETECTOR_H