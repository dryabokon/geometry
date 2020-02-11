import numpy
import cv2
from scipy.spatial import Delaunay
import tools_IO
import tools_image
import tools_GL3D
import tools_aruco
import pyrr
# ---------------------------------------------------------------------------------------------------------------------
import time
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
use_camera = False
camera_W, camera_H = 640, 480
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global list_filenames,filename_actor
    global image_actor

    if key==ord('a') or key==ord('d'):
        idx = tools_IO.smart_index(list_filenames, filename_actor)[0]
        if key==ord('d'):
            idx  =(idx+1)%len(list_filenames)
        else:
            idx = (idx-1+len(list_filenames)) % len(list_filenames)
        filename_actor = list_filenames[idx]

        image_actor = cv2.imread(folder_in + filename_actor)
        image_actor = tools_image.smart_resize(image_actor, camera_H, camera_W)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live():

    global image_actor
    #R = tools_GL3D.render_GL()
    #R.load_obj('./images/ex_GL/box.obj')

    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(3, camera_W)
        cap.set(4, camera_H)
    else:
        cap = None


    cnt, start_time, fps = 0, time.time(), 0
    while (True):
        if use_camera:
            ret, image_actor = cap.read()
            image_actor = cv2.flip(image_actor, 1)

        L = D.get_landmarks(image_actor)
        if D.are_landmarks_valid(L):
            del_triangles = Delaunay(L).vertices
            D.draw_landmarks_v2(image_actor,L,del_triangles)
            result = D.draw_landmarks(image_actor)
            r_vec, t_vec = D.get_pose(L)
            result = D.draw_annotation_box(result,r_vec, t_vec)
            #result = R.get_image(result)
            #result = R.render_obj()
            #result = R.morph_3D_mesh(camera_H,camera_W,result,r_vec, t_vec)
        else:
            result = image_actor.copy()

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
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
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    filename_actor = list_filenames[0]
    image_actor = cv2.imread(folder_in + filename_actor)
    image_actor = tools_image.smart_resize(image_actor, camera_H, camera_W)
    #demo_live()



    frame = cv2.imread('images/ex_aruco/02.jpg')
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box.obj',W=camera_W,H=camera_H)
    #frame, rvec, tvec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length=0.1, camera_matrix=R.cameraMatrix, dist=numpy.zeros(4))


    rvec, tvec = (0,0.8,0),(0,0,+3)

    R.init_mat_projection0()
    image = R.get_image((0,0,0), tvec)
    cv2.imwrite('./images/output/res0.png',image)

    R.init_mat_projection()
    image = R.get_image((0, 0, 0), tvec)
    cv2.imwrite('./images/output/res.png', image)





