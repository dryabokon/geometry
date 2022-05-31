import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
from CV import tools_vanishing
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
VP = tools_vanishing.detector_VP(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def BEV_van_point3(filename_in, cam_fov_deg,point_van_xy_ver,point_van_xy_hor=None,do_rotation=True):

    image = tools_image.desaturate(cv2.imread(filename_in), level=0)
    h_ipersp,target_BEV_W, target_BEV_H,rot_deg = VP.get_inverce_perspective_mat_v3(image, cam_fov_deg,point_van_xy_ver,point_van_xy_hor)
    image_BEV = cv2.warpPerspective(image, h_ipersp, (target_BEV_W, target_BEV_H), borderValue=(32, 32, 32))
    if do_rotation:
        image_BEV = tools_image.rotate_image(image_BEV, rot_deg, reshape=True, borderValue=(32, 32, 32))

    image_BEV = tools_image.auto_crop(image_BEV, background_color=(32, 32, 32))
    image_BEV = tools_image.do_resize(image_BEV,numpy.array((-1,image.shape[0])))

    #image_BEV = tools_draw_numpy.draw_text(image_BEV,'yaw %d deg'%rot_deg,(image_BEV.shape[1]-250,image_BEV.shape[0]-30),(0,0,200),(32,32,32),24)
    image_result = numpy.concatenate([image,image_BEV],axis=1)

    cv2.imwrite(folder_out + filename_in.split('/')[-1], image_result)

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

if __name__ == '__main__':

    # BEV_van_point3('./images/ex_BEV/_0000000000.png', cam_fov_deg=90,point_van_xy_ver = (631, 166))
    # BEV_lines('./images/ex_BEV/KZ.jpg', (330,764,1062,38), (924,1054,1346,62),42)
    # BEV_lines('./images/ex_BEV/MELCO_loc_train_03.jpg', (236,438,996,1052), (1810,780,236,226),23)
    # BEV_lines('./images/ex_BEV/loc_train_01_00001.jpg', (9,813,752,420), (1080,870,1240,328),42)
    # BEV_lines('./images/ex_BEV/Abu_dhabi1.jpg', (181, 86, 935, 382), (9, 516, 67, 700), cam_fov_deg=30)
    # BEV_lines('./images/ex_BEV/image7.jpg', (581, 271, 626, 660), (770, 316, 1040, 675), cam_fov_deg=44)

    #BEV_lines('./images/ex_BEV/screenshot_008.png', (636, 219, 636, 450), (880, 219, 1069, 611), 45)
    #BEV_lines('./images/ex_BEV/screenshot_013.png', (635, 358, 635, 660), (1098, 358, 1275, 658), 90)
    #BEV_lines('./images/ex_BEV/screenshot_023.png',  cam_fov_deg=26.26)


    BEV_van_point3('./images/ex_BEV/TNO_7180R_20220418135426.jpg', cam_fov_deg=7, point_van_xy_ver=(144,-526), point_van_xy_hor=(733392,45799), do_rotation=True)


