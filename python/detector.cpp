#include "detector.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <LandmarkCoreIncludes.h>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <Face_utils.h>
//#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

using std::cout;
using std::string;
using std::vector;
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

    Utilities::Visualizer visualizer(true, false, false, false);
	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
//	if (detection_certainty < visualisation_boundary)
//	{


		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);

        visualizer.SetImage(captured_image, fx, fy, cx, cy);
	    visualizer.SetObservationLandmarks(face_model.detected_landmarks, detection_certainty, face_model.GetVisibilities());
		visualizer.SetObservationPose(pose_estimate_to_draw, detection_certainty);
		visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy), detection_certainty);
        char character_press = visualizer.ShowObservation();

			// restart the tracker
			if (character_press == 'r')
			{
				std::cout << "pressed r";
			}

//	}

//	// Work out the framerate
//	if (frame_count % 10 == 0)
//	{
//		double t1 = cv::getTickCount();
//		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
//		t0 = t1;
//	}

	// Write out the framerate on the image before displaying it
//	char fpsC[255];
//	std::sprintf(fpsC, "%d", (int)fps_tracker);
//	string fpsSt("FPS:");
//	fpsSt += fpsC;
//	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
//
//	if (!det_parameters.quiet_mode)
//	{
//		cv::namedWindow("tracking_result", 1);
//		cv::imshow("tracking_result", captured_image);
//	}
}


Detector * Detector::Create(const char *binary_path) {

  vector<string> arguments;
  arguments.push_back(string("  ")); // if CPP this is the application name
  arguments.push_back(string("-mloc"));
  arguments.push_back(string(binary_path));

  LandmarkDetector::FaceModelParameters det_parameters(arguments);
  // No need to validate detections, as we're not doing tracking.
  det_parameters.validate_detections = false;
//  det_parameters.track_gaze = true;
  // Grab camera parameters, if they are not defined 
  // (approximate values will be used).
//  float fx = 0, fy = 0, cx = 0, cy = 0;
//  int device = -1;
//  // Get camera parameters
//  LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);
//  // If cx (optical axis centre) is undefined will use the image size/2 as
//  // an estimate.
//  bool cx_undefined = false;
//  bool fx_undefined = false;
//  if (cx == 0 || cy == 0) {
//    cx_undefined = true;
//  }
//  if (fx == 0 || fy == 0) {
//    fx_undefined = true;
//  }

  // The modules that are being used for tracking.
  LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
  
  cv::CascadeClassifier classifier(det_parameters.face_detector_location);
  dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

  return new Detector(det_parameters, clnf_model, classifier, face_detector_hog);

}

Detector::Detector(LandmarkDetector::FaceModelParameters &det_parameters,
                   LandmarkDetector::CLNF &clnf_model,
                   cv::CascadeClassifier &classifier,
                   dlib::frontal_face_detector &face_detector_hog) : det_parameters_(std::move(det_parameters)), clnf_model_(std::move(clnf_model)),
                                                                     classifier_(std::move(classifier)),
                                                                     face_detector_hog_(std::move(face_detector_hog)) {}

bool CompareRect(cv::Rect_<double> r1, cv::Rect_<double> r2) {

  return r1.height < r2.height;

}


bool Detector::getLandmarksInImage(cv::Mat &grayscale_frame, cv::Rect_<double> &face_rect) {
        return LandmarkDetector::DetectLandmarksInImage(grayscale_frame, face_rect, clnf_model_, det_parameters_);
}

void Detector::detectGaze(cv::Mat &grayscale_frame, cv::Point3f &gazeDirection0,cv::Point3f &gazeDirection1,bool externalDetection, bool debug, float fx, float fy, float cx, float cy) {

			bool detection_success;

			if(!externalDetection)
			{
				detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_frame, clnf_model_, det_parameters_);
			}
//			else
//			{
//				detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_frame, clnf_model_, det_parameters_);
//			}
			if ( (detection_success || externalDetection) && clnf_model_.eye_model)
			{
				GazeAnalysis::EstimateGaze(clnf_model_, gazeDirection0, fx, fy, cx, cy, true);
				GazeAnalysis::EstimateGaze(clnf_model_, gazeDirection1, fx, fy, cx, cy, false);
			}

            if(debug){
                visualise_tracking(grayscale_frame, clnf_model_, det_parameters_, gazeDirection0, gazeDirection1, 0, fx, fy, cx, cy);
            }


}

cv::Vec6d Detector::detectHeadPose(cv::Mat &grayscale_frame, bool use_world_coordinates, bool externalDetection, float fx, float fy, float cx, float cy) {
			bool detection_success;
            if(!externalDetection)
			{
				detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_frame, clnf_model_, det_parameters_);
			}
            cv::Vec6d pose_estimate;
            if(detection_success || externalDetection){
                if(use_world_coordinates)
                {
                    pose_estimate = LandmarkDetector::GetPose(clnf_model_, fx, fy, cx, cy);
                }
                else
                {
                    pose_estimate = LandmarkDetector::GetPoseWRTCamera(clnf_model_, fx, fy, cx, cy);
			    }
			}

			return pose_estimate;

}






