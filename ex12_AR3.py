import cv2
import numpy
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_wavefront
import tools_IO
import tools_render_GL
from CV import tools_calibrator
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
W, H = 1280, 720
#W,H = 1920,1080
# ----------------------------------------------------------------------------------------------------------------------
# def define_obj_position(filename_in=None):
#     Calibrator = tools_calibrator.Calibrator()
#     if filename_in is None:
#         flat_obj = (-1, -1, -1, +1, +1, +1)
#         filename_in = folder_out+'temp.obj'
#         points_3d = Calibrator.construct_cuboid_v0(flat_obj)
#         Calibrator.save_obj_file(filename_in,flat_obj)
#     else:
#         object = tools_wavefront.ObjLoader()
#         object.load_mesh(filename_in,do_autoscale=False)
#         points_3d = object.coord_vert
#
#     points_3d = tools_pr_geom.apply_rotation((0,0,0),points_3d)
#     points_3d = tools_pr_geom.apply_translation((-2, 0, +5), points_3d)
#
#     return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def example_project_GL_vs_CV(folder_out, filename_in=None):

    cam_fov = 11
    cam_offset_dist = 64
    cam_height = 6.5
    a_pitch = -numpy.arctan(cam_height / cam_offset_dist)

    empty = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
    camera_matrix_3x3, rvec, tvec, RT_GL = tools_render_GL.define_cam_position(W,H,cam_fov,cam_offset_dist,cam_height,a_pitch)
    tg_half_fovx = camera_matrix_3x3[0,2]/camera_matrix_3x3[0,0]
    textured = False#textured = (filename_in is not None)

    R = tools_GL3D.render_GL3D(filename_obj=filename_in,do_normalize_model_file=False,W=W, H=H,is_visible=False,projection_type='P',textured=textured,
                               rvec =(0,0,5*numpy.pi/180),tvec=(-2,0,-5))
    points_3d = R.object.coord_vert

    cv2.imwrite(folder_out + 'GL_image_perspective_V1.png', R.get_image_perspective(rvec, tvec, tg_half_fovx, tg_half_fovx, do_debug=True, mat_view_to_1=True ))
    cv2.imwrite(folder_out + 'GL_image_perspective_V0.png', R.get_image_perspective(rvec, tvec, tg_half_fovx, tg_half_fovx, do_debug=True, mat_view_to_1=False))
    cv2.imwrite(folder_out + 'GL_image_perspective_M.png' , R.get_image_perspective_M(RT_GL   , tg_half_fovx, tg_half_fovx, do_debug=True))
    cv2.imwrite(folder_out + 'GL_draw_cube_MVP.png'      , tools_render_GL.draw_cube_MVP_GL  (points_3d, empty, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'GL_draw_pnts_MVP.png'      , tools_render_GL.draw_points_MVP_GL(points_3d, empty, R.mat_projection, R.mat_view, R.mat_model, R.mat_trns))
    cv2.imwrite(folder_out + 'GL_draw_pnts_RT.png'       , tools_render_GL.draw_points_RT_GL (points_3d, empty, RT_GL, camera_matrix_3x3, R.mat_trns))
    cv2.imwrite(folder_out + 'GL_draw_pnts_rvec_tvec.png', tools_render_GL.draw_points_rvec_tvec_GL(points_3d, empty, rvec, tvec, camera_matrix_3x3, R.mat_trns))
    cv2.imwrite(folder_out + 'CV_draw_cube_rvec_tvec.png', tools_render_GL.draw_cube_rvec_tvec_GL  (points_3d, empty, rvec, tvec, camera_matrix_3x3, R.mat_trns))
    cv2.imwrite(folder_out + 'CV_cuboids_RT.png'         , tools_draw_numpy.draw_cuboid(empty,tools_render_GL.project_points_RT_GL(points_3d,RT_GL,camera_matrix_3x3,R.mat_trns)   ,color=(0, 90, 255), w=2,idx_mode=2))
    cv2.imwrite(folder_out + 'CV_cuboids_rvec_tvec.png'  , tools_draw_numpy.draw_cuboid(empty,tools_render_GL.project_points_rvec_tvec_GL(points_3d, rvec,tvec, camera_matrix_3x3,R.mat_trns),color=(0,190, 255), w=2))

    return
# ----------------------------------------------------------------------------------------------------------------------
filename_box= './images/ex_GL/box/box_2.obj'
filename_box_aligned= './images/ex_GL/box/box_2_aligned.obj'
filename_car = './images/ex_GL/car/SUV1.obj'
#filename_car= './images/ex_GL/box/temp.obj'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # object = tools_wavefront.ObjLoader()
    # object.convert(filename_car_raw, filename_car_aligned)

    tools_IO.remove_files(folder_out)
    example_project_GL_vs_CV(folder_out,filename_car)