import cv2
import dlib
import os
from scipy.spatial import Delaunay
import imutils
from imutils.face_utils import FaceAligner
# ---------------------------------------------------------------------------------------------------------------------
import time
import tools_IO
import detector_landmarks
import tools_image
import tools_faceswap
# ---------------------------------------------------------------------------------------------------------------------
use_camera = True
camera_W, camera_H = 640, 480
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(folder_out):

    filenames = tools_IO.get_filenames(folder_out,'*.jpg')
    if len(filenames)>0:
        filenames.sort()
        idx = int(filenames[-1].split('.')[0])
    else:
        idx =0

    use_camera = True
    cnt, start_time, fps = 0, time.time(), 0

    cap = cv2.VideoCapture(0)
    cap.set(3, camera_W)
    cap.set(4, camera_H)
    while (True):
        if use_camera:
            ret, image = cap.read()
            L = D.get_landmarks(image)
            if D.are_frontface_landmarks(L):
                result = D.cut_face(image, L, target_W=camera_W, target_H=camera_H)
                landmarks = D.get_landmarks(result)
                position_dist_hor = D.get_position_distance_hor(result,landmarks)
                position_dist_ver = D.get_position_distance_ver(result, landmarks)
                if position_dist_hor<0.1 and position_dist_ver <0.1:
                    cv2.imwrite(folder_out+'%06d.jpg'%idx,result)
                    idx = idx + 1
            else:
                position_dist_hor,position_dist_ver=1,1
                result = image.copy()

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, '{0: 1.2f} {1: 1.2f}'.format(position_dist_hor,position_dist_ver), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame', result)
        cnt+= 1

        key = cv2.waitKey(1)
        if key & 0xFF == 27:break

    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    folder_in = './images/ex_faceswap/01/'
    folder_out =  './images/output/'
    filename_base = 'Person1g.jpg'
    image_base = cv2.imread(folder_in + filename_base)
    #FS = tools_faceswap.Face_Swaper(D, image_base,image_base,device='cpu',adjust_every_frame=False,do_narrow_face=False)


    demo_live(folder_out)





