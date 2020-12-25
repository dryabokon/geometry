import math
import cv2
import numpy
import tools_GL3D
import pyrr
import tools_pr_geom
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_aruco
import tools_render_CV
import tools_image
import detector_landmarks
import tools_wavefront
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
def example_project_GL_vs_CV_acuro():
    marker_length = 1
    aperture_x,aperture_y = 0.5, 0.5

    frame = cv2.imread('./images/ex_aruco/01.jpg')
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box.obj', W=frame.shape[1], H=frame.shape[0],is_visible=False,projection_type='P')

    mat_camera = tools_pr_geom.compose_projection_mat_3x3(frame.shape[1], frame.shape[0],aperture_x,aperture_y)

    axes_image, r_vec, t_vec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, mat_camera, numpy.zeros(4))
    cv2.imwrite('./images/output/cube_CV.png', tools_render_CV.draw_cube_numpy(axes_image, mat_camera, numpy.zeros(4), r_vec.flatten(), t_vec.flatten(), (0.5,0.5,0.5)))

    image_3d = R.get_image_perspective(r_vec.flatten(), t_vec.flatten(),aperture_x,aperture_y,scale=(0.5,0.5,0.5),do_debug=True)
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    cv2.imwrite('./images/output/cube_GL.png', tools_image.blend_avg(frame, image_3d, clr, weight=0))

    r_vec, t_vec = r_vec.flatten(), t_vec.flatten()
    print('[ %1.2f, %1.2f, %1.2f], [%1.2f,  %1.2f,  %1.2f],  %1.2f' % (r_vec[0], r_vec[1], r_vec[2], t_vec[0], t_vec[1], t_vec[2],aperture_x))
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_project_GL_vs_CV(filename_in, folder_out):

    tools_IO.remove_files(folder_out)

    W, H = 800, 800
    empty = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
    rvec, tvec , aperture = (0.0, 0, 0), [+5.0, 0, +15],0.50
    rvec = numpy.array(rvec)
    tvec = numpy.array(tvec)
    scale = 1.0


    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H,is_visible=False,projection_type='P')
    RT_GL = tools_pr_geom.compose_RT_mat(rvec, tvec,do_flip=True)
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H, aperture, aperture)

    cv2.imwrite(folder_out + 'GL.png',    R.get_image_perspective  (rvec, tvec,aperture,aperture,scale=scale*numpy.array((1,1,1)),lookback=False,do_debug=True))
    cv2.imwrite(folder_out + 'GL_RT.png', R.get_image_perspective_M(RT_GL ,aperture,aperture,scale=scale*numpy.array((1,1,1)),lookback=False,do_debug=True))

    object = tools_wavefront.ObjLoader()
    object.load_mesh(filename_in, do_autoscale=True)
    points_3d = numpy.array(object.coord_vert,dtype=numpy.float32)

    mat_trans = scale * numpy.eye(4)
    mat_trans[3,3]=1

    mat_flip = numpy.eye(4)
    mat_flip[0,0]*=-1

    RT_CV = tools_pr_geom.compose_RT_mat(rvec, tvec, do_flip=False)

    cv2.imwrite(folder_out + 'CV_MVP_points.png', tools_render_CV.draw_points_numpy_MVP(points_3d, empty, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'CV_RT.png'        , tools_render_CV.draw_points_numpy_RT (points_3d, empty,RT_CV,camera_matrix_3x3))
    cv2.imwrite(folder_out + 'CV_numpy.png'     , tools_render_CV.draw_cube_numpy(empty, camera_matrix_3x3, numpy.zeros(4), rvec, tvec))
    cv2.imwrite(folder_out + 'CV_MVP_cube.png'  , tools_render_CV.draw_cube_numpy_MVP(empty, R.mat_projection,mat_flip.dot(R.mat_view), R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'CV_cuboids_M.png' , tools_draw_numpy.draw_cuboid(empty,tools_pr_geom.project_points_M(points_3d,RT_CV, camera_matrix_3x3, numpy.zeros(5)), color=(0, 90, 255), w=2))
    cv2.imwrite(folder_out + 'CV_cuboids.png'   , tools_draw_numpy.draw_cuboid(empty,tools_pr_geom.project_points  (points_3d, rvec,tvec,camera_matrix_3x3,numpy.zeros(5))[0],color=(0,190, 255), w=2))

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
    result = tools_render_CV.draw_points_numpy_MVP(L3D, image_actor, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
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
    W, H = 800, 800
    empty = numpy.full((H,W,3),32,dtype=numpy.uint8 )
    rvec, tvec = numpy.array((0, 0.1, 0)), numpy.array((0, 1, 5))
    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H, is_visible=False, do_normalize_model_file=True,projection_type='P')
    object = tools_wavefront.ObjLoader()
    object.load_mesh(filename_in, do_autoscale=True)
    points_3d = object.coord_vert

    mat_camera = tools_pr_geom.compose_projection_mat_3x3(W, H)

    cv2.imwrite(folder_out + 'cube_GL.png', R.get_image_perspective(rvec, tvec))
    cv2.imwrite(folder_out + 'cube_CV.png',tools_render_CV.draw_cube_numpy(empty, mat_camera, numpy.zeros(4), rvec,tvec))
    cv2.imwrite(folder_out + 'cube_CV_MVP.png', tools_render_CV.draw_cube_numpy_MVP  (           numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'points_MVP.png', tools_render_CV.draw_points_numpy_MVP(points_3d, numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))

    point_2d = (W-500,500)
    ray_begin, ray_end = tools_render_CV.get_ray(point_2d, mat_camera, R.mat_view, R.mat_model, R.mat_trns)
    ray_inter1 = tools_render_CV.line_plane_intersection((1, 0, 0), (-1, 0, 1), ray_begin - ray_end, ray_begin)
    ray_inter2 = tools_render_CV.line_plane_intersection((1, 0, 0), (+1, 0, 1), ray_begin - ray_end, ray_begin)
    points_3d_ray = numpy.array([ray_begin,ray_end,ray_inter1,ray_inter2])

    result = tools_draw_numpy.draw_ellipse(empty,((W-point_2d[0]-10,point_2d[1]-10,W-point_2d[0]+10,point_2d[1]+10)),(0,0,90))
    result = tools_render_CV.draw_points_numpy_MVP(points_3d_ray,result,R.mat_projection, R.mat_view, R.mat_model, R.mat_trns,color=(255, 255, 255))
    cv2.imwrite(folder_out +  'points_MVP.png', result)


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_ray_interception(filename_in):
    W, H = 800, 800
    rvec, tvec = (0, 0, 0), (0, 0, 5)
    point_2d = (W - 266, 266)
    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H, is_visible=False, do_normalize_model_file=True,projection_type='P')
    cv2.imwrite('./images/output/cube_GL.png', R.get_image_perspective(rvec, tvec))
    mat_camera_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H)

    ray_begin, ray_end= tools_render_CV.get_ray(point_2d, mat_camera_3x3,R.mat_view, R.mat_model, R.mat_trns)

    triangle = numpy.array([[+1, -2, -2], [+1, -2, 2], [+1, +2, 2]])
    collision = tools_render_CV.get_interception_ray_triangle(ray_begin, ray_end - ray_begin, triangle)
    print(collision)
    return
# ----------------------------------------------------------------------------------------------------------------------
filename_box= './images/ex_GL/box/box_1.obj'
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

    #example_project_GL_vs_CV_acuro()
    example_project_GL_vs_CV(filename_box,'./images/output/')

    #example_face_ortho(filename_actor = './images/ex_faceswap/01/person1a.jpg',filename_obj= filename_head_obj1, filename_3dmarkers = filename_markers1)
    #example_face_ortho(filename_actor = './images/ex_faceswap/01/person1a.jpg',filename_obj= filename_head_obj3_cut, filename_3dmarkers = filename_markers3)

    #example_ray(filename_box,'./images/output/')