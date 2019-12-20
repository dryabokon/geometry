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
    cnt, start_time, fps = 0, time.time(), 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 640 // 2)
    cap.set(4, 480 // 2)
    while (True):
        if use_camera:
            ret, image = cap.read()
            result = D.draw_landmarks(image)

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        cv2.imshow('frame', result)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:break
        cnt+=1


    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    demo_live()


