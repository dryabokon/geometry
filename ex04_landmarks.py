import numpy
import cv2
from scipy.spatial import Delaunay
import tools_IO
import tools_image
import tools_GL3D
# ---------------------------------------------------------------------------------------------------------------------
import time
import detector_landmarks
import tools_render_CV
import tools_wavefront
# ---------------------------------------------------------------------------------------------------------------------
capturing_devices = ['cam','mp4','image']
#camera_W, camera_H = 640, 480
camera_W, camera_H = 912, 1024
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global list_filenames,filename_actor
    global image_actor_default

    if key in [ord('a'),ord('d'),ord('w'),ord('s')]:
        idx = tools_IO.smart_index(list_filenames, filename_actor)[0]
        if key in [ord('d'),ord('w')]:
            idx  =(idx+1)%len(list_filenames)
        else:
            idx = (idx-1+len(list_filenames)) % len(list_filenames)
        filename_actor = list_filenames[idx]

        image_actor_default = cv2.imread(folder_in + filename_actor)
        image_actor_default = tools_image.smart_resize(image_actor_default, camera_H, camera_W)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(filename_obj,filename_3dmarkers=None):

    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_3dmarkers)

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=camera_W, H=camera_H,is_visible=False,do_normalize_model_file=True)
    R.transform_model('xz')

    if capturing_device == 'cam':
        cap = cv2.VideoCapture(0)
        cap.set(3, camera_W)
        cap.set(4, camera_H)
    elif capturing_device == 'mp4':
        cap = cv2.VideoCapture(filename_actor)

    cnt, start_time, fps = 0, time.time(), 0
    while (True):

        if capturing_device == 'image':
            image_actor = image_actor_default.copy()
        else:
            ret, image_actor = cap.read()
            if capturing_device=='mp4':
                xxx = tools_image.smart_resize(image_actor,camera_H,camera_W)
                cv2.imwrite('./images/output/A.jpg',xxx)
                image_actor = cv2.imread('./images/output/A.jpg')
            if capturing_device=='cam':
                cv2.flip(image_actor, 1)

        L = D.get_landmarks(image_actor)
        if D.are_landmarks_valid(L):
            del_triangles = Delaunay(L).vertices
            result = D.draw_landmarks_v2(image_actor,L)
            rvec, tvec  = D.get_pose(image_actor,L,D.model_68_points,R.mat_trns)
            result = tools_render_CV.draw_points_numpy_RT(D.model_68_points, result, R.mat_camera, numpy.zeros(4), rvec, tvec,R.mat_trns)

            #result = D.draw_annotation_box(result,r_vec, t_vec)
            image_3d = R.get_image(rvec,tvec)
            clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
            result = tools_image.blend_avg(image_actor, image_3d, clr, weight=0)
            print(rvec, tvec)

        else:
            result = image_actor.copy()

        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', result)

        cnt += 1
        key = cv2.waitKey(1)
        process_key(key)
        if key & 0xFF == 27: break

    if capturing_device == 'cam':cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
filename_head_obj1 = './images/ex_GL/face/face.obj'
filename_markers1 ='./images/ex_GL/face/markers_face.txt'
# ----------------------------------------------------------------------------------------------------------------------
filename_head_obj2 = './images/ex_GL/face/head.obj'
filename_markers2 ='./images/ex_GL/face/markers_head.txt'
# ----------------------------------------------------------------------------------------------------------------------
filename_head_obj3 = './images/ex_GL/face/head_scaled.obj'
filename_markers3 ='./images/ex_GL/face/markers_head_scaled.txt'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    capturing_device = 'image'
    folder_in = './images/ex_faceswap/01/'
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    filename_actor = list_filenames[0]

    image_actor_default = cv2.imread(folder_in+filename_actor)
    image_actor_default = tools_image.smart_resize(image_actor_default, camera_H, camera_W)

    demo_live(filename_head_obj3,filename_markers3)

