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
default_filename_in2  = './images/ex_faceswap/01/person1.jpg'
default_filename_in = './images/ex_faceswap/dm.jpg'
default_folder_out  = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def do_average(L2_original_hist):

    th = 100
    nose_x = L2_original_hist[:,30,0]
    nose_y = L2_original_hist[:,30,1]
    #stdx = numpy.std(nose_x)
    #stdy = numpy.std(nose_y)
    if True:# stdx<th  and stdy < th:
        avgx = numpy.average(L2_original_hist[:,:,0],0)
        avgy = numpy.average(L2_original_hist[:,:,1],0)

        res = numpy.vstack((avgx,avgy)).T
    else:
        res = L2_original_hist[0]


    return res
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(filename_out):

    image1 = cv2.imread('./images/ex_faceswap/01/person5.jpg')
    L1_original = D.get_landmarks(image1)

    hist = 5
    L2_original_hist = numpy.zeros((hist, L1_original.shape[0], 2))

    cap = cv2.VideoCapture(0)
    #cap.set(3, 320)
    #cap.set(4, 240)
    cnt, start_time, fps = 0, time.time(), 0
    while (True):

        ret, image2 = cap.read()
        image2 = cv2.flip(image2, 1)

        L2_original = D.get_landmarks(image2)
        L2_original_hist = numpy.roll(L2_original_hist, 1, 0)
        L2_original_hist[0] = L2_original
        L2_original = do_average(L2_original_hist)

        result = tools_landmark.do_reenackement(image1, image2, L1_original, L2_original)

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result2 = result.copy()
        result2 = cv2.putText(result2, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)

        w,h = 128, int(128 * image2.shape[0] / image2.shape[1])
        small = D.draw_landmarks(image2)
        small = cv2.resize(small,(w,h))
        result2[:h,-w:,:] = small


        cv2.imshow('frame', result2)
        cnt += 1
        key = cv2.waitKey(1)

        if key & 0xFF == 27:break
        if (key & 0xFF == 13) or (key & 0xFF == 32):
            cv2.imwrite(filename_out, image2)


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

    res2 = tools_landmark.do_reenackement(image1, image2, L1_original, L2_original)
    cv2.imwrite(default_folder_out + 'result2_v2.jpg', res2)

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    demo_auto_02()
    #demo_live(default_folder_out+'res.jpg')

