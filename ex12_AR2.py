import cv2
import numpy
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
import tools_aruco
import tools_render_CV
import tools_image
import detector_landmarks
import tools_wavefront
# ----------------------------------------------------------------------------------------------------------------------
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ----------------------------------------------------------------------------------------------------------------------
def example_project_GL_vs_CV_acuro():
    marker_length = 1

    frame = cv2.imread('./images/ex_aruco/01.jpg')
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box.obj', W=frame.shape[1], H=frame.shape[0],
                               is_visible=False,
                               do_transform_view=False,
                               scale=(0.5,0.5,0.5))

    axes_image, rvec, tvec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, R.mat_camera, numpy.zeros(4))

    cv2.imwrite('./images/output/cube_CV.png', tools_render_CV.draw_cube_numpy(frame, R.mat_camera, numpy.zeros(4), rvec.flatten(), tvec.flatten(), (0.5,0.5,0.5)))

    image_3d = R.get_image(rvec.flatten(), tvec.flatten())
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    cv2.imwrite('./images/output/cube_GL.png', tools_image.blend_avg(frame, image_3d, clr, weight=0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_project_GL_vs_CV(filename_in):

    W, H = 800, 800
    rvec, tvec = (0, 0, 0), (0, 0, 5)

    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H,is_visible=False,
                               do_transform_view=True)

    cv2.imwrite('./images/output/cube_GL.png', R.get_image(rvec, tvec))

    object = tools_wavefront.ObjLoader()
    object.load_mesh(filename_in, (215, 171, 151), do_autoscale=True)
    points_3d = object.coord_vert

    result = tools_render_CV.draw_points_numpy_MVP(points_3d, numpy.full((H,W,3),76,dtype=numpy.uint8), R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
    cv2.imwrite('./images/output/cube_CV_MVP.png', result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_face(filename_actor,filename_obj):

    image_actor = cv2.imread(filename_actor)
    #image_actor = tools_image.smart_resize(image_actor, 480, 640)

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=image_actor.shape[1], H=image_actor.shape[0],is_visible=False,
                               do_transform_view=True,
                               do_normalize_model_file=False)

    L = D.get_landmarks(image_actor)

    rvec, tvec = D.get_pose(image_actor,L)

    L3D = D.model_68_points
    image_3d = R.get_image(rvec, tvec)
    clr = (255 * numpy.array(R.bg_color)).astype(numpy.int)
    result = tools_image.blend_avg(image_actor, image_3d, clr, weight=0)
    cv2.imwrite('./images/output/face_GL.png', result)

    result = tools_render_CV.draw_points_numpy_RT(L3D, image_actor, R.mat_camera, numpy.zeros(4), rvec, tvec)
    cv2.imwrite('./images/output/face_CV_RT.png', result)

    result = tools_render_CV.draw_points_numpy_MVP(L3D, image_actor, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
    cv2.imwrite('./images/output/face_CV_MVP.png', result)


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_ray(filename_in):
    W, H = 800, 800
    rvec, tvec = (0, 0, 0), (0, 0, 5)
    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H, is_visible=False, do_normalize=True)
    object = tools_wavefront.ObjLoader()
    object.load_mesh(filename_in, (215, 171, 151), do_autoscale=True)
    points_3d = object.coord_vert

    cv2.imwrite('./images/output/cube_GL.png', R.get_image(rvec, tvec))

    result  = tools_render_CV.draw_cube_numpy_MVP  (           numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
    cv2.imwrite('./images/output/image_CV_cube.png', result)

    result = tools_render_CV.draw_points_numpy_MVP(points_3d, numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
    cv2.imwrite('./images/output/image_CV_cube_points.png', result)

    point_2d = (W-266,266)
    ray_begin, ray_end = tools_render_CV.get_ray(point_2d, numpy.full((H,W,3),76,dtype=numpy.uint8), R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)

    ray_inter1 = tools_render_CV.line_plane_intersection((1, 0, 0), (-1, 0, 1), ray_begin - ray_end, ray_begin)
    ray_inter2 = tools_render_CV.line_plane_intersection((1, 0, 0), (+1, 0, 1), ray_begin - ray_end, ray_begin)
    points_3d_ray = numpy.array([ray_begin,ray_end,ray_inter1,ray_inter2])

    result = tools_render_CV.draw_points_numpy_MVP(points_3d_ray, numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns,color=(66, 66, 0))
    cv2.imwrite('./images/output/image_CV_point1.png', result)


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_ray_interception(filename_in):
    W, H = 800, 800
    rvec, tvec = (0, 0, 0), (0, 0, 5)
    point_2d = (W - 266, 266)
    R = tools_GL3D.render_GL3D(filename_obj=filename_in, W=W, H=H, is_visible=False, do_normalize=True)
    cv2.imwrite('./images/output/cube_GL.png', R.get_image(rvec, tvec))

    ray_begin, ray_end= tools_render_CV.get_ray(point_2d, numpy.full((H, W, 3), 76, dtype=numpy.uint8), R.mat_projection,R.mat_view, R.mat_model, R.mat_trns)

    triangle = numpy.array([[+1, -2, -2], [+1, -2, 2], [+1, +2, 2]])
    collision = tools_render_CV.get_interception_ray_triangle(ray_begin, ray_end - ray_begin, triangle)
    print(collision)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #filename_head = './images/ex_GL/mesh.obj'
    filename_head = './images/ex_GL/face/face.obj'




    #example_project_GL_vs_CV_acuro()
    #example_project_GL_vs_CV(filename_cat)
    example_face(filename_actor = './images/ex_faceswap/01/person1a.jpg',filename_obj= filename_head)