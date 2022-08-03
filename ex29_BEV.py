import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
from CV import tools_vanishing
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
VP = tools_vanishing.detector_VP(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def BEV_van_point3(filename_in, fov_x_deg,point_van_xy_ver,point_van_xy_hor=None,do_rotation=False):

    image = cv2.imread(filename_in)
    fov_y_deg = fov_x_deg*image.shape[0]/image.shape[1]
    VP.H, VP.W = image.shape[:2]
    image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = VP.build_BEV_by_fov_van_point(image , fov_x_deg, fov_y_deg, point_van_xy_ver, point_van_xy_hor, do_rotation=do_rotation)
    #image_BEV, df_keypoints_pitch, df_vertical = VP.draw_grid_at_BEV(image_BEV, p_camera_BEV_xy, p_center_BEV_xy,lines_edges, fov_x_deg, fov_y_deg)

    cv2.imwrite(folder_out + filename_in.split('/')[-1], image_BEV)

    # image_BEV = tools_image.auto_crop(image_BEV, background_color=(32, 32, 32))
    # image_BEV = tools_image.do_resize(image_BEV,numpy.array((-1,image.shape[0])))
    # image_result = numpy.concatenate([image,image_BEV],axis=1)
    # cv2.imwrite(folder_out + filename_in.split('/')[-1], image_result)

    return

# ----------------------------------------------------------------------------------------------------------------------
def BEV_lines(filename_in,cam_fov_deg = 90.0):

    image = cv2.imread(filename_in)
    VP.H,VP.W = image.shape[:2]
    lines_ver = VP.get_lines_ver_candidates_single_image(image,do_debug=True)
    vp_ver, lines_vp_ver = VP.get_vp(VP.reshape_lines_as_paired(lines_ver))
    #fov_y_deg = cam_fov_deg * image.shape[0] / image.shape[1]
    #image_BEV_proxy, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV, lines_edges = VP.build_BEV_by_fov_van_point(image, vp_ver, cam_fov_deg, fov_y_deg, do_rotation=True)
    #cv2.imwrite(folder_out + filename_in.split('/')[-1], image_BEV_proxy)
    BEV_van_point3(filename_in, cam_fov_deg,vp_ver)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipeline_BEV(folder_in):
    filenames = tools_IO.get_filenames(folder_in,'*.jpg')
    fov_x_deg = 75
    point_van_xy_ver = (1000, 540)
    point_van_xy_hor = tools_render_CV.line_intersection((1194,588,224,628), (954,708,34,710))
    dist = -0.85

    for filename_in in filenames[2700:]:
        image = cv2.imread(folder_in+filename_in)
        fov_y_deg = fov_x_deg * image.shape[0] / image.shape[1]
        VP.H, VP.W = image.shape[:2]
        camera_matrix = numpy.array([[VP.W, 0., VP.W / 2], [0., VP.H, VP.H / 2], [0., 0., 1.]])
        image = cv2.undistort(image, camera_matrix, dist, None, None)
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = VP.build_BEV_by_fov_van_point(image, fov_x_deg, fov_y_deg, point_van_xy_ver, point_van_xy_hor, do_rotation=do_rotation)

        image_BEV = image_BEV[-640:,1715-200:1715+200]#cv2.imwrite(folder_out + filename_in.split('/')[-1],image_BEV)
        image_BEV = tools_image.auto_crop(image_BEV, background_color=(32, 32, 32))
        image_BEV = tools_image.do_resize(image_BEV,numpy.array((-1,image.shape[0])))
        image_result = numpy.concatenate([image,image_BEV],axis=1)
        cv2.imwrite(folder_out + filename_in.split('/')[-1], image_result)

    return
# ----------------------------------------------------------------------------------------------------------------------
#point_van_xy_hor = tools_render_CV.line_intersection((536,617,1246,588), (1522,697,517,726))
point_van_xy_hor = tools_render_CV.line_intersection((1194,588,224,628), (954,708,34,710))
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    do_rotation = False

    #BEV_van_point3('D://ccc/02727.jpg', fov_x_deg=75, point_van_xy_ver=(1012, 528),point_van_xy_hor=point_van_xy_hor,do_rotation=do_rotation)
    #BEV_van_point3('./images/ex_BEV/CityHealth/00000u.jpg', fov_x_deg=80, point_van_xy_ver=(1012, 528),point_van_xy_hor=point_van_xy_hor,do_rotation=do_rotation)
    #BEV_van_point3('./images/ex_BEV/CityHealth/00000.jpg', fov_x_deg=80, point_van_xy_ver=(1000, 540),point_van_xy_hor=point_van_xy_hor,do_rotation=do_rotation)
    # BEV_van_point3('./images/ex_BEV/0000000000.png', fov_x_deg=90,point_van_xy_ver = (631, 166),do_rotation=do_rotation)
    # BEV_van_point3('./images/ex_BEV/TNO_7180R_20220418135426.jpg', fov_x_deg=9, point_van_xy_ver=(144,-526), point_van_xy_hor=(733392,45799), do_rotation=do_rotation)
    # BEV_van_point3('./images/ex_BEV/TNO-7180R_20220525174045.jpg', fov_x_deg=9, point_van_xy_ver=(-391,-935), point_van_xy_hor=(6755,-46), do_rotation=do_rotation)
    # BEV_van_point3('./images/ex_BEV/TNO-7180R_20220418134537.jpg', fov_x_deg=8.5, point_van_xy_ver=(1047, -1522),point_van_xy_hor=(-103164, 3660), do_rotation=do_rotation)

    #pipeline_BEV('./images/ex_BEV/CityHealth/')
    pipeline_BEV('D://ccc/')

