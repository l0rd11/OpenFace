#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <LandmarkCoreIncludes.h>
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
using std::unique_ptr;

class Detector {

public:
  static Detector * Create(const char *binary_path);

  void detectGaze(cv::Mat &grayscale_frame, cv::Point3f &gazeDirection0,cv::Point3f &gazeDirection1,bool externalDetection, bool debug, float fx, float fy, float cx, float cy);
  cv::Vec6d detectHeadPose(cv::Mat &grayscale_frame, bool use_world_coordinates, bool externalDetection, float fx, float fy, float cx, float cy);
  bool getLandmarksInImage(cv::Mat &grayscale_frame, cv::Rect_<double> &face_rect);
  cv::Mat_<uchar> grayscale_frame_;
  LandmarkDetector::CLNF clnf_model_;

private:
  Detector(LandmarkDetector::FaceModelParameters &det_parameters,
           LandmarkDetector::CLNF &clnf_model,
           cv::CascadeClassifier &classifier,
           dlib::frontal_face_detector &face_detector_hog);

  LandmarkDetector::FaceModelParameters det_parameters_;
  
  cv::CascadeClassifier classifier_;
  dlib::frontal_face_detector face_detector_hog_;

};
