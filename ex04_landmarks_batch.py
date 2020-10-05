import numpy
import cv2
import time
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_GL3D
import detector_landmarks
import tools_pr_geom
# ---------------------------------------------------------------------------------------------------------------------
folder_in = 'D:\\LM/ex02/'
folder_out = 'D:\\LM/output_girl2/'
# ---------------------------------------------------------------------------------------------------------------------
filename_head_obj1='./images/ex_GL/face/head.obj'
filename_markers1 = './images/ex_GL/face/markers_head.txt'
# ---------------------------------------------------------------------------------------------------------------------
#camera_W, camera_H = 1920, 1080
camera_W, camera_H = 930, 525
# ---------------------------------------------------------------------------------------------------------------------
R = tools_GL3D.render_GL3D(filename_obj=filename_head_obj1, W=camera_W, H=camera_H,is_visible=False, do_normalize_model_file=False)
# ---------------------------------------------------------------------------------------------------------------------
def lm_process(folder_in,folder_out):
    filenames = numpy.array(tools_IO.get_filenames(folder_in, '*.jpg,*.png'))
    tools_IO.remove_files(folder_out,create=True)
    clr = (255 * numpy.array(R.bg_color[:3])).astype(numpy.int)

    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_markers1)
    L_prev = None

    for filename_in in filenames:
        time_start = time.time()
        base_name, ext = filename_in.split('/')[-1].split('.')[0], filename_in.split('/')[-1].split('.')[1]
        image = cv2.imread(folder_in+filename_in)
        gray = tools_image.desaturate(image)
        L = D.get_landmarks(image)

        #if not numpy.all(L==0):L_prev = L.copy()
        #else:L = L_prev.copy()

        rvec, tvec = D.get_pose_perspective(image, L, D.model_68_points, R.mat_trns)
        image_3d = R.get_image_perspective(rvec, tvec,lookback=True,scale=(1,2,1))

        #RT = tools_pr_geom.compose_RT_mat(rvec, tvec,do_rodriges=True)
        #image_3d = R.get_image_perspective_M(RT, lookback=True, scale=(1, 2, 1))

        #image_3d = cv2.flip(image_3d,1)
        #gray = cv2.imread('D:\LM\out11/'+base_name+'.jpg')
        #gray = tools_image.do_resize(gray,(image_3d.shape[1],image_3d.shape[0]))


        result = D.draw_landmarks_v2(gray, L,color=(0, 0, 200),w=4)
        #result = D.draw_annotation_box(result, rvec, tvec, color=(0, 128, 255), w=3)

        #result = tools_image.blend_avg(gray, image_3d, clr, weight=0.25)
        cv2.imwrite(folder_out+base_name+'.png',result)
        print('%s : %1.2f sec' % (base_name, (time.time() - time_start)))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    lm_process(folder_in,folder_out)
    #tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'ani_crash2m.gif', mask='*.jpg',framerate=24, resize_H=1080//4, resize_W=1920//4)