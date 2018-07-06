
import cv2
import time

from pyOpenFaceStub import PyOpenFaceStub


debug = True




cap = cv2.VideoCapture(0)


def main():
    pyOpenFaceStub = PyOpenFaceStub()
    detector = pyOpenFaceStub.detector
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            time_0 = time.time()
            gaze = detector.getGaze(img,False,True)

            yaw, pitch = pyOpenFaceStub.getGazeAngle(gaze)
            print yaw, pitch
            print "gaze = "
            print gaze
            time_1 = time.time()
            print "detect take {:.4f} Second".format(time_1 - time_0)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
