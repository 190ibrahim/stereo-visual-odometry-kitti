//
// Created by igor cvisic on 17.12.2022
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>

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

    Mat P0,P1;  //projection matrices for left and right camera

  int sequence=9; //select KITTI sequence

  //use data from KITTI calib.txt file in sequences to set projection matrices (images are rectified!)
  if (sequence<3) {
    double p0[12] = {7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    double p1[12] = {7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
                     0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    P0 = cv::Mat(3, 4, CV_64F, p0);
    P1 = cv::Mat(3, 4, CV_64F, p1);
  }
  if (sequence==3) {
    double p0[12] = {7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    double p1[12] = {7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.875744000000e+02,
                     0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    P0 = cv::Mat(3, 4, CV_64F, p0);
    P1 = cv::Mat(3, 4, CV_64F, p1);
  }
  if (sequence>3) {
    double p0[12] = {7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    double p1[12] = {7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, -3.798145000000e+02,
                     0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
                     0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00};
    P0 = cv::Mat(3, 4, CV_64F, p0);
    P1 = cv::Mat(3, 4, CV_64F, p1);
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
  sprintf(filename, "../dataset/results/%.2d.txt",sequence);
  pResults = fopen (filename,"w");

  while (true) {

    //LOAD IMAGES

    //load left image
    sprintf(name, "../dataset/sequences/%.2d/image_0/%.6d.png",sequence,name_counter);
    image0 = cv::imread(name,cv::IMREAD_GRAYSCALE);
    if (image0.data==NULL ) {
      std::cout<<"could not read "<<name<<std::endl;
      break;
    }
    //load right image
    sprintf(name, "../dataset/sequences/%.2d/image_1/%.6d.png",sequence,name_counter);
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
