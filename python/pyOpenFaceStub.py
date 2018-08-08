import os
import pyopenface
from math import atan2, sqrt, acos


class PyOpenFaceStub():
    def __init__(self):
        base_path = os.path.realpath(__file__)
        base_path = base_path[:base_path.find('python')]
        self.detector = pyopenface.Detector(base_path + "lib/local/LandmarkDetector/model/main_clnf_general.txt")
        self.width = 640.0
        self.height = 480.0
        self.multi = 500.0

    def getGazeAngle(self, gaze):
        gaze_vector = ((gaze[0] + gaze[3]) / 2.0, (gaze[1] + gaze[4]) / 2.0, (gaze[2] + gaze[5]) / 2.0)
        l = sqrt(gaze_vector[0] * gaze_vector[0] + gaze_vector[2] * gaze_vector[2])
        x_angle = atan2(gaze_vector[0], -gaze_vector[2])
        # y_angle = atan2(gaze_vector[1], -gaze_vector[2])
        # y_angle = (acos(gaze_vector[1]) - 1.5707963268)
        y_angle = (acos(gaze_vector[1] / l) - 1.5707963268)
        # y_angle = atan(gaze_vector[1] / sqrt(pow(gaze_vector[0], 2) + pow(gaze_vector[2], 2)))
        return x_angle, y_angle

    def getGaze(self,image, externalDetector, debug):
        cx = image.shape[1] / 2.0
        cy = image.shape[0] / 2.0
        fx = self.multi * (image.shape[1] / self.width)
        fy = self.multi * (image.shape[0] / self.height)
        fx = (fx + fy) / 2.0
        fy = fx
        return self.detector.getGaze(image, externalDetector, debug, fx, fy, cx, cy)

    def getHeadPose(self, image, use_world_coordinates, externalDetector):
        cx = image.shape[1] / 2.0
        cy = image.shape[0] / 2.0
        fx = self.multi * (image.shape[1] / self.width)
        fy = self.multi * (image.shape[0] / self.height)
        fx = (fx + fy) / 2.0
        fy = fx
        return self.detector.getHeadPose(image, use_world_coordinates, externalDetector, fx, fy, cx, cy)

    def getLandmarksInImage(self, image, rect):
        return self.detector.getLandmarksInImage(image, rect)

    def reset(self):
        self.detector.doReset()


