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
import tools_landmark
import detector_landmarks
import tools_animation
import tools_GL
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
use_camera = True
do_transfer = True
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global filename_clbrt, filename_actor
    global image_clbrt,image_actor
    global use_camera,do_transfer
    global L_actor,L_clbrt
    global result
    global del_triangles_C
    global R_c,R_a
    global folder_in

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
        cv2.imwrite(folder_out+'C.jpg', image_clbrt)
        cv2.imwrite(folder_out+'A.jpg', image_actor)
        cv2.imwrite(folder_out + 'R.jpg', result)


    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live():

    global filename_clbrt, filename_actor
    global image_clbrt,image_actor
    global use_camera,do_transfer
    global L_actor,L_clbrt
    global result
    global del_triangles_C
    global R_c, R_a

    camera_W, camera_H = 640, 480
    window_W, window_H = 640, 480

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
            L_actor = D.get_landmarks_augm(image_actor)
            L_actor[:,0]*= window_W/camera_W
            L_actor[:,1]*= window_H/camera_H
            image_actor = cv2.resize(image_actor, (window_W, window_H))

            R_a.update_texture(image_actor)

        if do_transfer:
            result = tools_landmark.do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C)
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
def demo_auto_01(folder_out):

    global filename_clbrt, filename_actor
    global image_clbrt, image_actor
    global use_camera, do_transfer
    global L_actor, L_clbrt
    global del_triangles_C
    global R_c, R_a

    result = tools_landmark.do_faceswap(R_c, R_a, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C, folder_out=folder_out, do_debug=True)
    cv2.imwrite(folder_out + 'result.jpg' , result)
    return
# ---------------------------------------------------------------------------------------------------------------------
def init(folder_in):

    global filename_clbrt, filename_actor
    global image_clbrt,image_actor
    global use_camera,do_transfer
    global L_actor,L_clbrt
    global del_triangles_C
    global R_c,R_a

    filename_clbrt, filename_actor = list_filenames[0], list_filenames[-1]
    image_clbrt = cv2.imread(folder_in + filename_clbrt)
    image_actor = cv2.imread(folder_in + filename_actor)
    if image_clbrt is None:
        print('%s not found' % (folder_in + filename_clbrt))
        exit()
    if image_actor is None:
        print('%s not found' % (folder_in + filename_actor))
        exit()
    L_clbrt = D.get_landmarks_augm(image_clbrt)
    if L_clbrt.min() == L_clbrt.max() == 0:
        print('Landmarks for clbrt image not found')
        exit()
    del_triangles_C = Delaunay(L_clbrt).vertices
    L_actor = D.get_landmarks_augm(image_actor)
    R_c = tools_GL.render_GL(image_clbrt)
    R_a = tools_GL.render_GL(image_actor)
    return
# ---------------------------------------------------------------------------------------------------------------------
def main(folder_in,folder_out):

    init(folder_in)
    demo_auto_01(folder_out)
    #demo_live()
    return
# ---------------------------------------------------------------------------------------------------------------------
#tools_landmark.process_folder_extract_landmarks(D, 'D:/3/', folder_out, write_images=False, write_annotation=True)
#tools_landmark.interpolate(folder_out+'Landmarks.txt',folder_out+'Landmarks_filtered.txt')
#tools_landmark.filter_landmarks(folder_out+'Landmarks.txt',folder_out+'Landmarks_filtered.txt')
#tools_landmark.process_folder_draw_landmarks(D, 'D:/4/',[folder_out+'Landmarks.txt'], folder_out, delim='\t')
#tools_landmark.process_folder_faceswap_by_landmarks(D, folder_in+filename_clbrt,'D:/3/', folder_out+'Landmarks.txt', folder_out)
#demo_live()
#tools_animation.folder_to_video(folder_out,'D:/ani_full.mp4',mask='*.jpg',resize_W=1920//2,resize_H=960//2)
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':










    folder_in = './images/ex_faceswap/01/'
    folder_out = './images/output/'
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    list_filenames = ['Person1a.jpg','Person2g.jpg']

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='')
    parser.add_argument('--folder_in', default=folder_in)
    parser.add_argument('--folder_out', default=folder_out)
    args = parser.parse_args()
    main(args.folder_in,args.folder_out)

    #filename_clbrt = 'Person1a.jpg'
    #tools_landmark.process_folder_faceswap_by_landmarks(D, folder_in+filename_clbrt,'D:/4/', folder_out+'Landmarks1.txt', folder_out)
    #tools_animation.folder_to_video(folder_out, 'D:/ani_v07.mp4', mask='*.jpg', resize_W=1920 // 2, resize_H=960 // 2)
