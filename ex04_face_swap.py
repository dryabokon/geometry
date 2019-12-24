import time
import numpy
import cv2
import os
import argparse
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
# ---------------------------------------------------------------------------------------------------------------------
default_filename_C = './images/ex_faceswap/01/person1.jpg'
default_filename_A = './images/ex_faceswap/01/person2.jpg'
default_folder_out  = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
default_folder_in = './images/ex_faceswap/01/'
list_filenames   = ['person1.jpg','person2.jpg','person3.jpg','person4.jpg','person5.jpg','person6.jpg']
#list_filenames    = ['frame001685.jpg','frame001694.jpg','frame001700.jpg','frame001720.jpg']
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(folder_in):

    display_width, display_height = 640,480

    use_camera = True
    do_transfer = True

    filename_clbrt=list_filenames[0]
    filename_actor=list_filenames[1]

    image_clbrt = cv2.imread(folder_in+filename_clbrt)
    image_actor = cv2.imread(folder_in+filename_actor)
    image_actor = tools_image.smart_resize(image_actor,display_height,display_width)

    L_clbrt = D.get_landmarks(image_clbrt)
    del_triangles = Delaunay(L_clbrt).vertices
    L_actor = D.get_landmarks(image_actor)

    R_c = tools_GL.render_GL(image_clbrt)
    R_a = tools_GL.render_GL(image_actor)

    result = tools_landmark.do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles)
    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(3, display_width)
        cap.set(4, display_height)

    cnt, start_time, fps = 0, time.time(), 0
    while (True):
        if use_camera:
            ret, image_actor = cap.read()
            image_actor = tools_image.smart_resize(image_actor, display_height, display_width)
            L_actor = D.get_landmarks(image_actor)
            R_a.update_texture(image_actor)

            if do_transfer:
                result = tools_landmark.do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles)
            else:
                result = image_actor

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result2 = result.copy()

        result2 = cv2.putText(result2, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        result2 = cv2.putText(result2, 'Clbrt: {0}'.format(filename_clbrt), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        result2 = cv2.putText(result2, 'Actor: {0}'.format(filename_actor), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1,cv2.LINE_AA)

        cv2.imshow('frame', result2)
        cnt += 1
        key = cv2.waitKey(1)

        if key & 0xFF == 27:break
        if key & 0xFF == ord('1'): filename_clbrt = list_filenames[0]
        if key & 0xFF == ord('2'): filename_clbrt = list_filenames[1]
        if key & 0xFF == ord('3'): filename_clbrt = list_filenames[2]
        if key & 0xFF == ord('4'): filename_clbrt = list_filenames[3]
        if key & 0xFF == ord('5'): filename_clbrt = list_filenames[4]
        if key & 0xFF == ord('6'): filename_clbrt = list_filenames[5]
        if key & 0xFF == ord('7'): filename_clbrt = list_filenames[6]
        if key & 0xFF == ord('8'): filename_clbrt = list_filenames[7]

        if key & 0xFF == ord('0') or key & 0xFF == ord('`'):
            if use_camera:
                do_transfer = not do_transfer

        if (key & 0xFF >= ord('1')) and (key & 0xFF <= ord('9')):
            do_transfer = True
            image_clbrt = cv2.imread(folder_in + filename_clbrt)
            L_clbrt = D.get_landmarks(image_clbrt)
            del_triangles = Delaunay(L_clbrt).vertices
            R_c.update_texture(image_clbrt)

            result = tools_landmark.do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles)

        if key & 0xFF == ord('q'): filename_actor = list_filenames[0]
        if key & 0xFF == ord('w'): filename_actor = list_filenames[1]
        if key & 0xFF == ord('e'): filename_actor = list_filenames[2]
        if key & 0xFF == ord('r'): filename_actor = list_filenames[3]
        if key & 0xFF == ord('t'): filename_actor = list_filenames[4]
        if key & 0xFF == ord('y'): filename_actor = list_filenames[5]
        if key & 0xFF == ord('u'): filename_actor = list_filenames[6]
        if key & 0xFF == ord('i'): filename_actor = list_filenames[7]
        if key & 0xFF == ord('-'): use_camera = True;filename_actor='cam'
        if (key & 0xFF >= ord('a')) and (key & 0xFF <= ord('z')):
            use_camera = False
            image_actor = cv2.imread(folder_in + filename_actor)
            image_actor = tools_image.smart_resize(image_actor, display_height, display_width)
            L_actor = D.get_landmarks(image_actor)
            R_a.update_texture(image_actor)
            result = tools_landmark.do_transfer(R_c,R_a,image_clbrt, image_actor, L_clbrt, L_actor, del_triangles)

        if (key & 0xFF == 13) or (key & 0xFF == 32):
            cv2.imwrite('C.jpg', image_clbrt)
            cv2.imwrite('A.jpg', image_actor)
            tools_landmark.transferface_first_to_second(D, 'C.jpg', 'A.jpg', default_folder_out)

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_auto_01():

    res2 = tools_landmark.transferface_first_to_second(D, default_filename_C , default_filename_A, default_folder_out)
    cv2.imwrite(default_folder_out + 'first.jpg' , res2)
    #tools_landmark.morph_first_to_second(D,default_filename_in2, default_filename_in,default_folder_out,numpy.arange(0.1,0.9,0.1))
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #demo_auto_01()
    demo_live(default_folder_in)

