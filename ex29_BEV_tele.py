import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_render_CV
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def build_BEV(image,point_van_xy = (631, 166)):
    H, W = image.shape[:2]
    target_W,target_H = W,W


    tol_up = 10
    tol_bottom = 0
    upper_line = (0, point_van_xy[1] + tol_up, W, point_van_xy[1] + tol_up)
    bottom_line = (0, H - tol_bottom, W, H - tol_bottom)

    pad_left = 2 * W
    pad_right = 2 * W
    line1 = (-pad_left, H, point_van_xy[0], point_van_xy[1])
    line2 = (W + pad_right, H, point_van_xy[0], point_van_xy[1])
    p1 = tools_render_CV.line_intersection(upper_line, line1)
    p2 = tools_render_CV.line_intersection(upper_line, line2)
    p3 = tools_render_CV.line_intersection(bottom_line, line1)
    p4 = tools_render_CV.line_intersection(bottom_line, line2)
    src = numpy.array([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])], dtype=numpy.float32)
    dst = numpy.array([(0, 0), (target_W, 0), (0, target_H), (target_W, target_H)], dtype=numpy.float32)
    #image = tools_draw_numpy.draw_convex_hull(image, [p1, p2, p3, p4], color=(36, 10, 48), transperency=0.5)

    h_ipersp = cv2.getPerspectiveTransform(src, dst)
    image_BEV = cv2.warpPerspective(image, h_ipersp, (target_W, target_H), borderValue=(0, 0, 0))
    return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files('./images/output/')

    image = tools_image.desaturate(cv2.imread('./images/ex_BEV/0000000038.png'))
    image_BEV = build_BEV(image)
    cv2.imwrite('./images/output/BEV.png', image_BEV)
