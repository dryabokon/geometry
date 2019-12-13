import numpy
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
default_filename_in  = './images/ex_faceswap/01/personC-1.jpg'
default_filename_in2 = './images/ex_faceswap/01/personD-1.jpg'
default_folder_in    = './images/ex_faceswap/02/'
default_folder_out  = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(filename_out):

    image1 = cv2.imread('./images/ex_faceswap/01/personD-1.jpg')
    L1_original = D.get_landmarks(image1)
    idx_mouth = D.idx_head + D.idx_nose + D.idx_eyes
    del_triangles = Delaunay(L1_original).vertices
    del_triangles_mouth = Delaunay(L1_original[idx_mouth]).vertices


    cap = cv2.VideoCapture(0)
    cnt, start_time, fps = 0, time.time(), 0
    while (True):

        ret, image2 = cap.read()
        L2_original = D.get_landmarks(image2)
        result = tools_landmark.do_reenackement(image1, image2, L1_original, L2_original,idx_mouth,del_triangles,del_triangles_mouth)

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result2 = result.copy()
        result2 = cv2.putText(result2, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)

        w,h = 128, int(128 * image2.shape[0] / image2.shape[1])
        image2 = cv2.resize(image2,(w,h))
        result2[:h,:w,:] = image2
        result2 = cv2.flip(result2,1)

        cv2.imshow('frame', result2)
        cnt += 1
        key = cv2.waitKey(1)

        if key & 0xFF == 27:break

    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_auto_02():

    res2 = tools_landmark.transfer_emo(D, default_filename_in , default_filename_in2,default_folder_out)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_auto_03():


    image1 = cv2.imread(default_filename_in2)
    image2 = cv2.imread(default_filename_in)
    L1_original = D.get_landmarks(image1)
    L2_original = D.get_landmarks(image2)
    idx_mouth = D.idx_head + D.idx_nose + D.idx_eyes
    del_triangles = Delaunay(L1_original).vertices
    del_triangles_mouth = Delaunay(L1_original[idx_mouth]).vertices

    res2 = tools_landmark.do_reenackement(image1, image2, L1_original, L2_original, idx_mouth, del_triangles,del_triangles_mouth)
    cv2.imwrite(default_folder_out + 'result2_v2.jpg', res2)

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #demo_auto_03()
    demo_live(default_folder_out+'res.jpg')

