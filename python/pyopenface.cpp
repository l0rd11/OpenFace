#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <opencv2/core.hpp>
#include "detector.hpp"


namespace bp = boost::python;


bp::list getGaze(Detector *detector, bp::object frame_obj, bool externalDetection, bool debug, float fx, float fy, float cx, float cy)
{

    PyArrayObject *frame_arr = reinterpret_cast<PyArrayObject *>(frame_obj.ptr());
    const int height = PyArray_DIMS(frame_arr)[0];
    const int width = PyArray_DIMS(frame_arr)[1];
    cv::Mat grayscale_frame(cv::Size(width, height),
		    CV_8UC1,
		    PyArray_DATA(frame_arr),
		    cv::Mat::AUTO_STEP);
    // Gaze tracking, absolute gaze direction

			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

    detector->detectGaze(grayscale_frame, gazeDirection0, gazeDirection1, externalDetection, debug, fx, fy, cx, cy);

    bp::list list;
    list.append(gazeDirection0.x);
    list.append(gazeDirection0.y);
    list.append(gazeDirection0.z);
    list.append(gazeDirection1.x);
    list.append(gazeDirection1.y);
    list.append(gazeDirection1.z);

    return list;

}

bp::list getHeadPose(Detector *detector, bp::object frame_obj,  bool useWorldCoordinates, bool externalDetection, float fx, float fy, float cx, float cy)
{

    PyArrayObject *frame_arr = reinterpret_cast<PyArrayObject *>(frame_obj.ptr());
    const int height = PyArray_DIMS(frame_arr)[0];
    const int width = PyArray_DIMS(frame_arr)[1];
    cv::Mat grayscale_frame(cv::Size(width, height),
		    CV_8UC1,
		    PyArray_DATA(frame_arr),
		    cv::Mat::AUTO_STEP);

    cv::Vec6d pose_estimate = detector->detectHeadPose(grayscale_frame, useWorldCoordinates, externalDetection, fx, fy, cx, cy);

    bp::list list;
    list.append(pose_estimate[0]);
    list.append(pose_estimate[1]);
    list.append(pose_estimate[2]);
    list.append(pose_estimate[3]);
    list.append(pose_estimate[4]);
    list.append(pose_estimate[5]);
    return list;

}


bp::object getLandmarksInImage(Detector *detector, bp::object frame_obj, const bp::list& rect_object)
{
     // list is face_rect[left, top, right, bottom]
    PyArrayObject *frame_arr = reinterpret_cast<PyArrayObject *>(frame_obj.ptr());
    const int height = PyArray_DIMS(frame_arr)[0];
    const int width = PyArray_DIMS(frame_arr)[1];
    cv::Mat grayscale_frame(cv::Size(width, height),
		    CV_8UC1,
		    PyArray_DATA(frame_arr),
		    cv::Mat::AUTO_STEP);

     bp::ssize_t rect_len = bp::len(rect_object);
    assert(rect_len==4);
    double rect_x = bp::extract<double>(rect_object[0]);
    double rect_y = bp::extract<double>(rect_object[1]);
    double rect_height = bp::extract<double>(rect_object[2]) - bp::extract<double>(rect_object[0]);
    double rect_width  = bp::extract<double>(rect_object[3]) - bp::extract<double>(rect_object[1]);
    cv::Rect_<double> face_rect( rect_x,
            rect_y,
            rect_height,
            rect_width);

    bool success = detector->getLandmarksInImage(grayscale_frame,face_rect);


    return bp::object(success);

}


void doReset(Detector *detector) {
    detector->reset();
}



BOOST_PYTHON_MODULE(pyopenface) {

  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
  
  bp::class_<Detector>("Detector", bp::no_init)
      .def("__init__", bp::make_constructor(&Detector::Create))
      .def("getGaze", &getGaze)
      .def("getHeadPose", &getHeadPose)
      .def("getLandmarksInImage", &getLandmarksInImage)
      .def("doReset", &doReset)
      ;


}
