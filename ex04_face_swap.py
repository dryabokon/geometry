import time
import numpy
import cv2
import os
from scipy.spatial import Delaunay
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import time
import tools_IO
import tools_landmark
import detector_landmarks
import tools_GL
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ----------------------------------------------------------------------------------------------------------------------
use_camera = False
do_transfer = True
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
folder_in = './images/ex_faceswap/01/'
#list_filenames   = ['person1.jpg','person2.jpg','person3.jpg','person4.jpg','person5.jpg','person6.jpg']
list_filenames   = ['person1a.jpg','person1c.jpg','person1c.jpg','person1d.jpg','person2a.jpg','person2b.jpg']
filename_clbrt,filename_actor = list_filenames[0],list_filenames[1]
image_clbrt = cv2.imread(folder_in + filename_clbrt)
image_actor = cv2.imread(folder_in + filename_actor)
L_clbrt = D.get_landmarks_augm(image_clbrt)
del_triangles_C = Delaunay(L_clbrt).vertices
L_actor = D.get_landmarks_augm(image_actor)
R_c = tools_GL.render_GL(image_clbrt)
R_a = tools_GL.render_GL(image_actor)
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global filename_clbrt, filename_actor
    global image_clbrt,image_actor
    global use_camera,do_transfer
    global L_actor,L_clbrt
    global del_triangles_C
    global R_c,R_a

    if key >= ord('1') and key <= ord('9') and key - ord('1')<len(list_filenames):
        filename_clbrt = list_filenames[key - ord('1')]
        do_transfer = True
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        L_clbrt = D.get_landmarks_augm(image_clbrt)
        del_triangles_C = Delaunay(L_clbrt).vertices
        R_c.update_texture(image_clbrt)

    lst = [ord('q'), ord('w'), ord('e'), ord('r'), ord('t'), ord('y')]
    if key in lst:
        idx = tools_IO.smart_index(lst, key)
        filename_actor = list_filenames[idx[0]]
        use_camera = False
        image_actor = cv2.imread(folder_in + filename_actor)
        L_actor = D.get_landmarks_augm(image_actor)
        R_a.update_texture(image_actor)

    if key==9:
        use_camera = not use_camera

    if key & 0xFF == ord('0') or key & 0xFF == ord('`'):
        if use_camera:
            do_transfer = not do_transfer

    if (key & 0xFF == 13) or (key & 0xFF == 32):
        cv2.imwrite('C.jpg', image_clbrt)
        cv2.imwrite('A.jpg', image_actor)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live():

    global filename_clbrt, filename_actor
    global image_clbrt,image_actor
    global use_camera,do_transfer
    global L_actor,L_clbrt
    global del_triangles_C
    global R_c, R_a

    camera_W, camera_H = 640, 480
    window_W, window_H = 640, 480

    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(3, camera_W)
        cap.set(4, camera_H)

    cnt, start_time, fps = 0, time.time(), 0
    while (True):
        if use_camera:
            ret, image_actor = cap.read()
            L_actor = D.get_landmarks_augm(image_actor)
            L_actor[:,0]*= window_W/camera_W
            L_actor[:,1]*= window_H/camera_H
            image_actor = cv2.resize(image_actor, (window_W, window_H))

            R_a.update_texture(image_actor)

        if do_transfer:
            result = tools_landmark.do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C)
        else:
            result = image_actor

        if time.time() > start_time: fps = cnt / (time.time() - start_time)


        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 1, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, 'Clbrt: {0}'.format(filename_clbrt), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 0, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, 'Actor: {0}'.format(filename_actor), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(1, 0, 0), 1,cv2.LINE_AA)

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
def demo_auto_01():

    res2 = tools_landmark.transferface_first_to_second(D, folder_in+filename_clbrt,folder_in+filename_actor, folder_out)
    cv2.imwrite(folder_in + 'first.jpg' , res2)
    #tools_landmark.morph_first_to_second(D,default_filename_in2, default_filename_in,default_folder_out,numpy.arange(0.1,0.9,0.1))
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    demo_auto_01()
    #demo_live()
    #tools_landmark.transferface_folder(D, folder_in+'person3a.jpg', 'D:/2/', folder_out)
    #tools_landmark.landmarks_folder(D, 'D:/2/', folder_out)

