import os
import pyopenface
from math import atan2


class PyOpenFaceStub():
    def __init__(self):
        base_path = os.path.realpath(__file__)
        base_path = base_path[:base_path.find('python')]
        self.detector = pyopenface.Detector(base_path + "lib/local/LandmarkDetector/model/main_clnf_general.txt")

    def getGazeAngle(self, gaze):
        gaze_vector = ((gaze[0] + gaze[3]) / 2, (gaze[1] + gaze[4]) / 2, (gaze[2] + gaze[5]) / 2)
        x_angle = atan2(gaze_vector[0], -gaze_vector[2])
        y_angle = atan2(gaze_vector[1], -gaze_vector[2])
        return x_angle, y_angle
