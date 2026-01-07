#include "feature_detector.h"

void detectFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                    const std::string& detector_type, int n_features) {
    cv::Ptr<cv::Feature2D> detector;
    if (detector_type == "ORB") {
        detector = cv::ORB::create(n_features);
    } 
    // else if (detector_type == "SIFT") {
    //     detector = cv::SIFT::create(n_features);
    // } 
    else {
        throw std::runtime_error("Unknown detector type: " + detector_type);
    }
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

void matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    matcher.match(descriptors1, descriptors2, matches);
}


void filterStereoMatches(
    const std::vector<cv::DMatch>& stereo_matches,
    const std::vector<cv::KeyPoint>& keypoints_left,
    const std::vector<cv::KeyPoint>& keypoints_right,
    std::vector<cv::Point2f>& matches0,
    std::vector<cv::Point2f>& matches1,
    std::vector<cv::Point2f>& fails)
{
    for (const auto& m : stereo_matches) {
        cv::Point2f pt_left = keypoints_left[m.queryIdx].pt;
        cv::Point2f pt_right = keypoints_right[m.trainIdx].pt;
        float epipolar_diff = std::abs(pt_left.y - pt_right.y);
        float disparity = pt_left.x - pt_right.x;
        if (epipolar_diff < 1.0 && disparity > 0) {
            matches0.push_back(pt_left);
            matches1.push_back(pt_right);
        } else {
            fails.push_back(pt_left);
        }
    }
    std::cout << "Stereo matches: " << stereo_matches.size() << std::endl;
    std::cout << "Good matches: " << matches0.size() << std::endl;
    std::cout << "Rejected matches: " << fails.size() << std::endl;
}