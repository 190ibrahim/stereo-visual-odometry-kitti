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



    int name_counter=0;
    char name[1024];

    Mat image0,image1;
    Mat image0_prev, image1_prev;
    Mat color0;

    //Current camera pose
    Mat Pose=Mat::eye(4,4,CV_64F);

    Scalar red(0,0,255);
    Scalar green(0,255,0);
    Scalar blue(255,0,0);

    bool pause=false;

    //open file for writing the results
    FILE *pResults = NULL;
    char filename[1024];
    snprintf(filename, 1024, "%s/results/%s.txt", dataset_root.c_str(), sequence_str.c_str());
    pResults = fopen (filename,"w");

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
      name_counter++;
      continue;
    }

    //DETECT FEATURES IN CURRENT LEFT IMAGE

    //FIND MATCHES TO PREVIOUS LEFT

    //FIND MATCHES TO CURRENT RIGHT

    //FILTER GOOD MATCHES - LESS PROBLEMS IN THE OPTIMIZATION PART
    vector<Point2f> matches0,matches1,matches0_prev,matches1_prev;
    vector<Point2f> fails;

      //CHECK CURRENT LEFT-RIGHT EPIPOLAR CONSTRAINT

      //CHECK IF DISPARITY IS POSITIVE

      //SELECT GOOD MATCHES FROM CORRECT EPIPOLAR AND DISPARITY

    //PROJECT CURRENT MATCHES TO 3D 
    
    vector<bool> inliers_green(matches0.size(),false);

    //PREPARE ROTATION MATRIX AND TRANSLATION VECTOR TO BE SET BY THE METHOD
    Mat R=Mat::eye(3,3,CV_64F);
    double tx=0;
    double ty=0;
    double tz=0;
  
    //MINIMIZE REPROJECTION ERROR OF 3D POINTS FROM THE CURRENT FRAME INTO 2D POINTS OF THE PREVIOUS FRAME USING PnP

    //Construct SE3 matrix from rotation matrix and translation vector
    Mat SE3=Mat::eye(4,4,CV_64F);
    R.copyTo(SE3(Rect(0,0,3,3)));
    SE3.at<double>(0,3)=tx;
    SE3.at<double>(1,3)=ty;
    SE3.at<double>(2,3)=tz;

    //Concatenate last transform to get current pose
    Pose*=SE3;

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
        cv::line(color0,matches0[i],matches0_prev[i],green,2);
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

    name_counter++;
    }

    if (pResults) {
        fclose (pResults);
        cout<<"results written to "<<filename<<endl;
    }

    return 0;
}
