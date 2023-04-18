import cv2
import numpy
import tools_GL3D
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_wavefront
import tools_image
import tools_draw_numpy
import tools_render_GL
from CV import tools_aruco
from CV import tools_pr_geom
import tools_render_CV
from detector import detector_landmarks
# ----------------------------------------------------------------------------------------------------------------------
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
def example_acuro_pose_estimation():
    marker_length = 1

    fov_x,fov_y = 0.7, 0.7

    frame = cv2.imread('./images/ex_aruco/01.jpg')
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(frame.shape[1], frame.shape[0],fov_x,fov_y)

    axes_image, r_vec, t_vec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, camera_matrix_3x3, numpy.zeros(4))
    r_vec, t_vec = r_vec.flatten(), t_vec.flatten()
    #cv2.imwrite('./images/output/pose.png',axes_image)

    M_obj = numpy.eye(4)
    M_obj[3,3]=2

    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box_2.obj', do_normalize_model_file=False,W=frame.shape[1], H=frame.shape[0], is_visible=False, projection_type='P',M_obj=M_obj)
    image_3d = R.get_image_perspective(r_vec,t_vec,fov_x,fov_y,scale=(0.5,0.5,0.5),do_debug=True)
    cv2.imwrite('./images/output/cube_GL.png', tools_image.blend_avg(frame, image_3d, (255 * numpy.array(R.bg_color)).astype(numpy.int32), weight=0))


    point_2d = numpy.array((323, 132))
    #point_2d = numpy.array((300, 100))
    #point_2d = numpy.array((253, 117))
    image_cube_CV = tools_render_GL.draw_cube_rvec_tvec_GL(None, frame, r_vec, t_vec, camera_matrix_3x3, scale=(0.5, 0.5, 0.5),color=(255, 128, 0), w=6)
    ray_begin, ray_end = tools_render_CV.get_ray(point_2d, frame, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns, d=10)
    point_3d = tools_render_CV.get_interception_ray_triangle(ray_begin, (ray_begin - ray_end), numpy.array([(100,0,-1),(-100,0,-1),(0,100,-1)]), allow_outside=True)
    #collisions_3d = tools_render_CV.get_interceptions_ray_triangles(ray_begin, (ray_begin - ray_end),R.object.coord_vert, R.object.idx_vertex)

    point_2dp, _ = tools_pr_geom.project_points(point_3d, r_vec, t_vec, camera_matrix_3x3)
    #point_2dp, _ = tools_pr_geom.project_points(collisions_3d, r_vec, t_vec, camera_matrix_3x3)
    image_cube_CV = tools_draw_numpy.draw_points(image_cube_CV, point_2dp, color=(0, 0, 200), w=4)
    cv2.imwrite('./images/output/cube_CV.png', image_cube_CV)
    ii=0
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_face_perspective(filename_actor,filename_obj,filename_3dmarkers=None,do_debug=False):

    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_3dmarkers)

    image_actor = cv2.imread(filename_actor)
    image_actor = tools_image.smart_resize(image_actor, 640, 640)

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=image_actor.shape[1], H=image_actor.shape[0],is_visible=False,projection_type='P',scale=(1,1,0.25))

    L = D.get_landmarks(image_actor)
    L3D = D.model_68_points
    L3D[:,2] = 0

    rvec, tvec= D.get_pose_perspective(image_actor,L,L3D,R.mat_trns)

    print('[ %1.2f, %1.2f, %1.2f], [%1.2f,  %1.2f,  %1.2f]'%(rvec[0],rvec[1],rvec[2],tvec[0],tvec[1],tvec[2]))

    image_3d = R.get_image_perspective(rvec, tvec,do_debug=do_debug)
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    result = tools_image.blend_avg(image_actor, image_3d, clr, weight=0)
    cv2.imwrite('./images/output/face_GL.png', result)

    M = pyrr.matrix44.multiply(pyrr.matrix44.create_from_eulers(rvec), pyrr.matrix44.create_from_translation(tvec))
    R.mat_model, R.mat_view = tools_pr_geom.decompose_model_view(M)
    result = tools_render_GL.draw_points_MVP_GL(L3D, image_actor, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
    result = D.draw_landmarks_v2(result, L)
    cv2.imwrite('./images/output/face_CV_MVP.png', result)


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_face_ortho(filename_actor,filename_obj,filename_3dmarkers=None):

    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_3dmarkers)
    image_actor = cv2.imread(filename_actor)

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=image_actor.shape[1], H=image_actor.shape[0],is_visible=False,projection_type='O',scale=(1, 1, 1))
    R.inverce_transform_model('Z')
    L = D.get_landmarks(image_actor)
    L3D = D.model_68_points

    rvec, tvec, scale_factor  = D.get_pose_ortho(image_actor,L,L3D,R.mat_trns,do_debug=True)
    print('[ %1.2f, %1.2f, %1.2f], [%1.2f,  %1.2f,  %1.2f], [%1.2f,%1.2f]'%(rvec[0],rvec[1],rvec[2],tvec[0],tvec[1],tvec[2],scale_factor[0],scale_factor[1]))

    image_3d = R.get_image_ortho(rvec, tvec, scale_factor, do_debug=True)
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    result = tools_image.blend_avg(image_actor, image_3d, clr, weight=0)
    cv2.imwrite('./images/output/face_GL_ortho.png', result)
    cv2.imwrite('./images/output/face_image_3d.png', image_3d)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_ray(filename_in, folder_out):
    W, H = 800, 600
    empty = numpy.full((H,W,3),32,dtype=numpy.uint8 )
    rvec_model, tvec_model = (0, 0, 0), (0, 0, 0)
    M_obj = tools_pr_geom.compose_RT_mat(rvec_model, tvec_model, do_rodriges=False, do_flip=False, GL_style=True)

    # d = +3.0
    # eye,target,up = (0, d, 0),(0, d-1, 0), (0, 0, 1)
    eye = (0,-10,0)
    target=(0,0,0)
    up=(0,0,1)

    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H, M_obj=M_obj,is_visible=False, do_normalize_model_file=False,projection_type='P',eye = eye,target=target,up=up)
    tg_half_fovx = 1/R.mat_projection[0][0]
    cv2.imwrite(folder_out + 'GL_image_perspective.png'  , R.get_image(do_debug=True))
    # cv2.imwrite(folder_out + 'GL_image_perspective_V1.png',R.get_image_perspective_M(numpy.eye(4), tg_half_fovx, tg_half_fovx, do_debug=True, mat_view_to_1=True))
    # cv2.imwrite(folder_out + 'GL_image_perspective_M1.png',R.get_image_perspective_M(numpy.eye(4), tg_half_fovx, tg_half_fovx, do_debug=True, mat_view_to_1=False))

    point_2d = numpy.array((W//2-40,H//2-30))
    ray_begin, ray_end = tools_render_CV.get_ray(point_2d,empty,R.mat_projection, R.mat_view, R.mat_model, R.mat_trns,d=10)

    collisions_3d = tools_render_CV.get_interceptions_ray_triangles(ray_begin, (ray_begin-ray_end), R.object.coord_vert,R.object.idx_vertex)
    collision_3d = R.get_best_collision(collisions_3d)
    points_3d = numpy.array([ray_begin,ray_end])
    if collision_3d is not None:
        points_3d = numpy.concatenate([points_3d,collision_3d.reshape((-1,3))])

    points_2d = tools_render_GL.project_points_MVP_GL(points_3d, W, H, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)

    cv2.imwrite(folder_out + 'CV_draw_pnts_numpy_MVP_GL.png', tools_render_CV.draw_points_numpy_MVP_GL(points_3d, empty, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns, color=(255, 255, 255)))
    cv2.imwrite(folder_out + 'GL_draw_pnts_MVP.png'         , tools_render_GL.draw_points_MVP_GL(points_3d, empty, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'points_ray.png', tools_draw_numpy.draw_points(empty, points_2d))
    cv2.imwrite(folder_out + 'points_orig.png', tools_draw_numpy.draw_points(empty, [point_2d]))

    return
# ----------------------------------------------------------------------------------------------------------------------
filename_box= './images/ex_GL/box/box_2.obj'
#filename_box= './images/ex_GL/box/temp.obj'
# ----------------------------------------------------------------------------------------------------------------------
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
#tools_render_CV.align_two_model(filename_head_obj2,filename_markers2,filename_head_obj1,filename_markers1,'./images/output/model.obj','./images/output/m.txt')
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_conversion('./images/output/', './images/ex_GL/box/box_1.obj')
    example_acuro_pose_estimation()
    #example_face_ortho(filename_actor = './images/ex_faceswap/01/person1a.jpg',filename_obj= filename_head_obj1, filename_3dmarkers = filename_markers1)
    #example_face_ortho(filename_actor = './images/ex_faceswap/01/person1a.jpg',filename_obj= filename_head_obj3_cut, filename_3dmarkers = filename_markers3)
    #example_ray(filename_box,'./images/output/')
