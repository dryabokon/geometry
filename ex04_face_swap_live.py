from scipy import signal
from scipy import ndimage
import sys
import numpy
import cv2
from scipy.spatial import Delaunay
import argparse
# ---------------------------------------------------------------------------------------------------------------------
import time
import tools_IO
import tools_faceswap
import detector_landmarks
import tools_animation
import tools_GL
import tools_animation
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
camera_W, camera_H = 640, 480
use_camera = False
do_transfer = True
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global use_camera,do_transfer
    global image_clbrt,image_actor
    global folder_in
    global list_filenames
    global filename_clbrt,filename_actor

    if key >= ord('1') and key <= ord('9') and key - ord('1')<len(list_filenames):
        filename_clbrt = list_filenames[key - ord('1')]
        do_transfer = True
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        FS.update_clbrt(image_clbrt)

    lst = [ord('q'), ord('w'), ord('e'), ord('r'), ord('t'), ord('y')]
    if key in lst:
        idx = tools_IO.smart_index(lst, key)
        filename_actor = list_filenames[idx[0]]
        use_camera = False
        image_actor = cv2.imread(folder_in + filename_actor)
        FS.update_actor(image_actor)
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        FS.update_clbrt(image_clbrt)

    if key==9:
        use_camera = not use_camera

    if key & 0xFF == ord('0') or key & 0xFF == ord('`'):
        if use_camera:
            do_transfer = not do_transfer

    if (key & 0xFF == 13) or (key & 0xFF == 32):
        cv2.imwrite(folder_out+'C.jpg', image_clbrt)
        cv2.imwrite(folder_out+'A.jpg', image_actor)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(FS):

    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(3, camera_W)
        cap.set(4, camera_H)
    else:
        cap = None

    cnt, start_time, fps = 0, time.time(), 0
    while (True):
        if use_camera:
            if cap is None:
                cap = cv2.VideoCapture(0)
                cap.set(3, camera_W)
                cap.set(4, camera_H)

            ret, image_actor = cap.read()
            FS.update_actor(image_actor)

        if do_transfer:
            result = FS.do_faceswap()
        else:
            result = image_actor

        if time.time() > start_time: fps = cnt / (time.time() - start_time)

        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 1, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, 'Clbrt: {0}'.format(filename_clbrt.split('/')[-1]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 0, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, 'Actor: {0}'.format(filename_actor.split('/')[-1]), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 0, 0), 1,cv2.LINE_AA)

        cv2.imshow('frame', result)
        cnt += 1
        key = cv2.waitKey(1)
        process_key(key)
        if key & 0xFF == 27: break

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = './images/ex_faceswap/01/'
    folder_out = './images/output1/'
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    filename_clbrt, filename_actor = folder_in + list_filenames[ 0], folder_in + list_filenames[ 1]
    image_clbrt = cv2.imread(filename_clbrt)
    image_actor = cv2.imread(filename_actor)

    FS = tools_faceswap.Face_Swaper(D, image_clbrt,image_actor)
    demo_live(FS)

