import cv2
import os
import argparse
from scipy.spatial import Delaunay
# ---------------------------------------------------------------------------------------------------------------------
import time
import tools_IO
import tools_landmark
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
def demo_live():

    use_camera = True

    cap = cv2.VideoCapture(0)
    while (True):
        if use_camera:
            ret, image = cap.read()
            result = D.draw_landmarks(image)

        cv2.imshow('frame', result)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:break


    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    demo_live()


