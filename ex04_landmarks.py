import numpy
import cv2
import tools_IO
import tools_image
import tools_GL3D
# ---------------------------------------------------------------------------------------------------------------------
import time
from detector import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
camera_W, camera_H = 640, 640
mode = 'L'#Box,L,Perspective
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global list_filenames,filename_actor
    global image_actor_default
    global mode

    if key in [ord('a'),ord('d'),ord('w'),ord('s')]:
        idx = tools_IO.smart_index(list_filenames, filename_actor)[0]
        if key in [ord('d'),ord('w')]:
            idx  =(idx+1)%len(list_filenames)
        else:
            idx = (idx-1+len(list_filenames)) % len(list_filenames)
        filename_actor = list_filenames[idx]

        image_actor_default = cv2.imread(folder_in + filename_actor)
        image_actor_default = tools_image.smart_resize(image_actor_default, camera_H, camera_W)


    if key == ord('1'): mode = 'L'
    #if key == ord('2'): mode = 'Box'
    if key == ord('3'): mode = 'Ortho'
    #if key == ord('4'): mode = 'Perspective'


    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(filename_obj,filename_3dmarkers,projection_type='O',scale=(1,1,1)):

    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_3dmarkers)
    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=camera_W, H=camera_H,is_visible=False, do_normalize_model_file=True,textured=False, projection_type=projection_type,scale=scale)

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
                cv2.imwrite('./images/output/A.jpg',tools_image.smart_resize(image_actor,camera_H,camera_W))
                image_actor = cv2.imread('./images/output/A.jpg')
                image_actor = tools_image.smart_resize(image_actor, camera_H, camera_W)
            if capturing_device=='cam':
                cv2.flip(image_actor, 1)
                image_actor = tools_image.smart_resize(image_actor, camera_H, camera_W)

        L = D.get_landmarks(image_actor)


        if D.are_landmarks_valid(L):
            if mode == 'L':
                #del_triangles = Delaunay(L).vertices
                result = D.draw_landmarks_v2(image_actor,L)

            if mode == 'Box':
                rvec, tvec   = D.get_pose_perspective(image_actor,L,D.model_68_points,R.mat_trns)
                result = D.draw_annotation_box(image_actor,rvec, tvec)

            if mode == 'Ortho':
                rvec, tvec, scale  = D.get_pose_ortho(image_actor,L,D.model_68_points,R.mat_trns)
                image_3d = R.get_image_ortho(rvec,tvec,scale)

                result = tools_image.blend_avg(image_actor, image_3d, (200,0,0), weight=0)

            if mode == 'Perspective':
                rvec, tvec  = D.get_pose_perspective(image_actor,L,D.model_68_points,R.mat_trns)
                image_3d = R.get_image_perspective(rvec,tvec)

                result = tools_image.blend_avg(image_actor, image_3d, (200,0,0), weight=0.5)


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
filename_head_obj3_cut = './images/ex_GL/face/head_scaled_cut.obj'
filename_markers3 ='./images/ex_GL/face/markers_head_scaled.txt'
# ----------------------------------------------------------------------------------------------------------------------
#filename_actor = './images/ex_DMS/JB_original.mp4'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    capturing_device = 'cam'

    # capturing_device = 'image'
    # folder_in = './images/ex_faceswap/01/'
    # list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    # filename_actor = list_filenames[0]
    # image_actor_default = cv2.imread(folder_in+filename_actor)
    # image_actor_default = tools_image.smart_resize(image_actor_default, camera_H, camera_W)

    #capturing_device = 'mp4'
    #filename_actor = 'D://ddd/'


    demo_live(filename_head_obj1,filename_markers1,'P')




