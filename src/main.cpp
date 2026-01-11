//
// Created by igor cvisic on 17.12.2022
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "feature_detector.h"
#include "pose_estimator.h"


using namespace std;
using namespace cv;

void savePoseKITTI(FILE *pFile, Mat Pose) {
  if (pFile && Pose.rows==4 && Pose.cols==4 && Pose.type()==CV_64F) {
    fprintf(pFile, "%e %e %e %e %e %e %e %e %e %e %e %e\r\n",
            Pose.at<double>(0,0),Pose.at<double>(0,1),Pose.at<double>(0,2),Pose.at<double>(0,3),
            Pose.at<double>(1,0),Pose.at<double>(1,1),Pose.at<double>(1,2),Pose.at<double>(1,3),
            Pose.at<double>(2,0),Pose.at<double>(2,1),Pose.at<double>(2,2),Pose.at<double>(2,3));
  }
}


int main(int argc, char** argv){
  // Load config.yaml
  std::string config_path = "../config.yaml";
  YAML::Node config = YAML::LoadFile(config_path);
  std::string dataset_root = config["dataset"]["root"].as<std::string>();
  std::string sequence_str = config["dataset"]["sequence"].as<std::string>();
  std::string detector_type = config["vo"]["detector"].as<std::string>();
  int sequence = std::stoi(sequence_str);


  Mat P0,P1;  //projection matrices for left and right camera

  // Build calib.txt path
  std::string calib_path = dataset_root + "/sequences/" + sequence_str + "/calib.txt";
  std::ifstream calib_file(calib_path);
  std::vector<std::vector<double>> proj(2, std::vector<double>(12, 0.0));
  if (calib_file.is_open()) {
    std::string line;
    int idx = 0;
    while (std::getline(calib_file, line) && idx < 2) {
      size_t pos = line.find(":");
  
      if (pos != std::string::npos) {
        std::istringstream iss(line.substr(pos + 1));
        for (int i = 0; i < 12; ++i) {
          iss >> proj[idx][i];
        }
      }
      idx++;
    }
    calib_file.close();
    P0 = cv::Mat(3, 4, CV_64F, proj[0].data()).clone();
    P1 = cv::Mat(3, 4, CV_64F, proj[1].data()).clone();
  } else {
    std::cerr << "Could not open calib.txt at " << calib_path << std::endl;
    return -1;
  }

  //camera matrix is the same for left and right camera
  Mat cameraMatrix = P0(cv::Rect(0,0,3,3)).clone();


  double f=P1.at<double>(0,0);
  double cu=P1.at<double>(0,2);
  double cv=P1.at<double>(1,2);
  double base=-P1.at<double>(0,3)/P1.at<double>(0,0);

  cout<<"focal range: "<<f<<endl;
  cout<<"principal point cu: "<<cu<<endl;
  cout<<"principal point cv: "<<cv<<endl;
  cout<<"baseline: "<<base<<endl;

  // Initialize feature detector
  StereoFeatureDetector feature_detector(detector_type);
  
  // Initialize stereo depth computation (like Python stereo_depth)
  StereoDepth stereo_depth(P0, P1);
  
  // Initialize motion estimator (like Python motion_estimation)
  MotionEstimator motion_estimator(cameraMatrix, 3000.0); // max_depth from config
  
  // Initialize ground truth loader (like Python DataLoader)
  GroundTruthLoader gt_loader(dataset_root, sequence_str);
  if (!gt_loader.loadGroundTruth()) {
    std::cerr << "Warning: Could not load ground truth for evaluation" << std::endl;
  }



    int name_counter=0;
    char name[1024];

    Mat image0,image1;
    Mat image0_prev, image1_prev;
    Mat color0;
    
    // Store previous frame keypoints and descriptors for matching
    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptors_prev;

    //Current camera pose
    Mat Pose=Mat::eye(4,4,CV_64F);

    Scalar red(0,0,255);
    Scalar green(0,255,0);
    Scalar blue(255,0,0);

    bool pause=false;

    //open file for writing the results
    FILE *pResults = NULL;
    FILE *pEvaluation = NULL;
    char filename[1024];
    char eval_filename[1024];
    snprintf(filename, 1024, "../results/%s.txt", sequence_str.c_str());
    snprintf(eval_filename, 1024, "../results/%s_evaluation.txt", sequence_str.c_str());
    pResults = fopen (filename,"w");
    pEvaluation = fopen(eval_filename, "w");
    
    // Write evaluation file header
    if (pEvaluation) {
        fprintf(pEvaluation, "frame,gt_x,gt_y,gt_z,est_x,est_y,est_z,trans_error,rot_error\n");
    }

  while (true) {

    //LOAD IMAGES


        //load left image
        snprintf(name, 1024, "%s/sequences/%s/image_0/%06d.png", dataset_root.c_str(), sequence_str.c_str(), name_counter);
        image0 = cv::imread(name,cv::IMREAD_GRAYSCALE);
        if (image0.data==NULL ) {
            std::cout<<"could not read "<<name<<std::endl;
            break;
        }
        //load right image
        snprintf(name, 1024, "%s/sequences/%s/image_1/%06d.png", dataset_root.c_str(), sequence_str.c_str(), name_counter);
        image1 = cv::imread(name,cv::IMREAD_GRAYSCALE);
        if (image1.data==NULL ) {
            std::cout<<"could not read "<<name<<std::endl;
            break;
        }

    if (name_counter==0) {
      //THE FIRST IMAGE, SAVE IDENTITY POSE AND CONTINUE
      savePoseKITTI(pResults,Pose);
      image0_prev=image0;
      image1_prev=image1;
      
      // Detect features in first frame for next iteration
      feature_detector.detectFeatures(image0, keypoints_prev, descriptors_prev);
      
      name_counter++;
      continue;
    }
    //DETECT FEATURES IN CURRENT LEFT IMAGE
    std::vector<cv::KeyPoint> keypoints_current;
    cv::Mat descriptors_current;
    feature_detector.detectFeatures(image0, keypoints_current, descriptors_current);

    //FIND MATCHES TO PREVIOUS LEFT
    std::vector<cv::DMatch> matches;
    if (!descriptors_prev.empty() && !descriptors_current.empty()) {
        feature_detector.matchFeatures(descriptors_prev, descriptors_current, matches);
    }

    //COMPUTE STEREO DEPTH (like Python stereo_depth function)
    cv::Mat depth_map = stereo_depth.computeDepth(image0, image1);
    std::cout << "Computed depth map: " << depth_map.size() << std::endl;
    
    //MOTION ESTIMATION USING PnP (like Python motion_estimation)
    cv::Mat R_est, t_est;
    bool motion_success = motion_estimator.estimateMotion(matches, keypoints_prev, keypoints_current, 
                                                         depth_map, R_est, t_est);
    
    if (motion_success) {
        // Create transformation matrix (like Python)
        cv::Mat T_est = cv::Mat::eye(4, 4, CV_64F);
        R_est.copyTo(T_est(cv::Rect(0, 0, 3, 3)));
        t_est.copyTo(T_est(cv::Rect(3, 0, 1, 3)));
        
        // Update pose (like Python: homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix)))
        Pose = Pose * T_est.inv();
    } else {
        std::cout << "Motion estimation failed, using identity transform" << std::endl;
    }
    
    // Progress reporting (like Python)
    if (name_counter % 10 == 0) {
        std::cout << "\n=== " << name_counter << " frames have been computed ===" << std::endl;
    }
    
    // Ground truth comparison (like Python visualization)
    if (gt_loader.getNumFrames() > name_counter) {
        gt_loader.printTrajectoryComparison(Pose, name_counter);
        
        // Save evaluation data to file
        if (pEvaluation) {
            cv::Point3f gt_pos = gt_loader.getGroundTruthPosition(name_counter);
            cv::Point3f est_pos(Pose.at<double>(0, 3), Pose.at<double>(1, 3), Pose.at<double>(2, 3));
            double trans_error = gt_loader.computeTranslationError(Pose, name_counter);
            double rot_error = gt_loader.computeRotationError(Pose, name_counter);
            
            fprintf(pEvaluation, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    name_counter, gt_pos.x, gt_pos.y, gt_pos.z, 
                    est_pos.x, est_pos.y, est_pos.z, trans_error, rot_error);
        }
    }

    //FILTER GOOD MATCHES - LESS PROBLEMS IN THE OPTIMIZATION PART
    vector<Point2f> matches0,matches1,matches0_prev,matches1_prev;
    vector<Point2f> fails;

    // Convert matches to Point2f for visualization (optional)
    for (const auto& match : matches) {
        if (motion_success) {
            matches0.push_back(keypoints_current[match.trainIdx].pt);
            matches0_prev.push_back(keypoints_prev[match.queryIdx].pt);
        }
    }

    //PROJECT CURRENT MATCHES TO 3D 
    
    vector<bool> inliers_green(matches0.size(), motion_success);

    // Motion estimation already computed R and t, no need for manual computation

    //Save current pose to file in a KITTI format
    savePoseKITTI(pResults,Pose);

    //DRAW MATCHES
    cvtColor(image0, color0, cv::COLOR_GRAY2BGR);
    snprintf(name, 1024, "3D-2D %d", name_counter);

    putText(color0, name, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX,1,blue,2);

    for (int i=0;i<fails.size();i++) {
      cv::circle(color0,fails[i],5,red);
    }

    for (int i=0;i<matches0.size();i++) {

      cv::circle(color0,matches0[i],3,blue);

      //cv::line(color0,matches0[i],matches1[i],blue);//draw disparity

      if (inliers_green[i]) {
        cv::line(color0,matches0[i],matches0_prev[i],green,1);
      } else {
        cv::line(color0,matches0[i],matches0_prev[i],red);
      }

      if (false) {
        snprintf(name,1024, "%d",i);
        putText(color0, name, matches0[i], cv::FONT_HERSHEY_SIMPLEX,0.5,red,1);
      }
    }

    imshow("IMAGE LEFT",color0);

    unsigned char key;
    if (pause) {
      key=cv::waitKey(1000000);
    } else {
      key=cv::waitKey(1);
    }
    if (key==' ') {
      pause=!pause;
    }

    image0_prev=image0;
    image1_prev=image1;
    
    // Store current frame data for next iteration
    keypoints_prev = keypoints_current;
    descriptors_prev = descriptors_current.clone();

    name_counter++;
    }

    if (pResults) {
        fclose (pResults);
        cout<<"results written to "<<filename<<endl;
    }
    
    if (pEvaluation) {
        fclose(pEvaluation);
        cout<<"evaluation data written to "<<eval_filename<<endl;
    }
    
    // Print final trajectory evaluation summary (like Python)
    if (gt_loader.getNumFrames() > 0) {
        std::cout << "\n=== FINAL TRAJECTORY EVALUATION ===" << std::endl;
        std::cout << "Processed " << name_counter << " frames" << std::endl;
        std::cout << "Ground truth available for " << gt_loader.getNumFrames() << " frames" << std::endl;
        
        // Compute average errors over last 50 frames
        double avg_trans_error = 0.0;
        double avg_rot_error = 0.0;
        int eval_frames = std::min(50, std::min(name_counter, gt_loader.getNumFrames()));
        
        for (int i = name_counter - eval_frames; i < name_counter && i < gt_loader.getNumFrames(); i++) {
            if (i >= 0) {
                avg_trans_error += gt_loader.computeTranslationError(Pose, i);
                avg_rot_error += gt_loader.computeRotationError(Pose, i);
            }
        }
        
        if (eval_frames > 0) {
            avg_trans_error /= eval_frames;
            avg_rot_error /= eval_frames;
            
            std::cout << "Average translation error (last " << eval_frames << " frames): " 
                      << std::fixed << std::setprecision(3) << avg_trans_error << " m" << std::endl;
            std::cout << "Average rotation error (last " << eval_frames << " frames): " 
                      << std::setprecision(2) << avg_rot_error << " degrees" << std::endl;
        }
    }

    return 0;
}
