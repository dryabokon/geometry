import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
filename_in = './images/ex_BEV/0000000038.png'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    image = tools_image.desaturate(cv2.imread(filename_in))
    point_van_xy = (631, 166)
    H, W = image.shape[:2]
    target_BEV_W, target_BEV_H = int(H * 0.75), H

    h_ipersp = tools_render_CV.get_inverce_perspective_mat_v2(image,target_BEV_W, target_BEV_H,point_van_xy)
    image_BEV = cv2.warpPerspective(image, h_ipersp, (target_BEV_W, target_BEV_H), borderValue=(32, 32, 32))
    image_result = numpy.zeros((H, W + target_BEV_W, 3), dtype=numpy.uint8)
    image_result[:, :W] = image
    image_result[:, W:] = image_BEV
    cv2.imwrite(folder_out + 'BEV.png', image_result)
