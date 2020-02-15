import cv2
import numpy
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
import tools_aruco
import tools_image
import detector_landmarks
# ----------------------------------------------------------------------------------------------------------------------
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ----------------------------------------------------------------------------------------------------------------------
def example_aruco():
    marker_length = 1
    frame = cv2.imread('./images/ex_aruco/01.jpg')
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box.obj', W=frame.shape[1], H=frame.shape[0],is_visible=False,do_normalize=False,scale=(0.5,0.5,0.01))
    axes_image, rvec, tvec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, R.mat_camera, numpy.zeros(4))

    result = tools_aruco.draw_cube_numpy(frame, R.mat_camera, numpy.zeros(4), rvec, tvec, (marker_length,marker_length,marker_length/100))
    cv2.imwrite('./images/output/cube_CV.png', result)

    image_3d = R.get_image(rvec.flatten(), tvec.flatten())
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    result = tools_image.blend_avg(frame, image_3d, clr, weight=0)
    cv2.imwrite('./images/output/cube_GL.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_face():
    marker_length = 1
    image_actor = cv2.imread('./images/ex_faceswap/01/person1a.jpg')
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/face/face.obj', W=image_actor.shape[1], H=image_actor.shape[0],is_visible=False,do_normalize=False)
    L = D.get_landmarks(image_actor)
    L3D = numpy.array(R.obj_list[0].coord_vert, dtype=numpy.float)
    rvec, tvec = D.get_pose(image_actor,L,L3D)



    result = tools_aruco.draw_mesh_numpy(L3D,image_actor, R.mat_camera, numpy.zeros(4), rvec, tvec, (marker_length,marker_length,marker_length))
    cv2.imwrite('./images/output/face_CV.png', result)

    image_3d = R.get_image(rvec, tvec)
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    result = tools_image.blend_avg(image_actor, image_3d, clr, weight=0)
    cv2.imwrite('./images/output/face_GL.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_face()
