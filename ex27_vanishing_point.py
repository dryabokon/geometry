#https://github.com/WolfgangFahl/play-chess-with-a-webcam/blob/master/examples/opencv/warp.py
import cv2
import numpy
from CV import tools_Hough
from CV import tools_Skeletone
from CV import tools_vanishing
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
Ske = tools_Skeletone.Skelenonizer(folder_out)
Hough = tools_Hough.Hough()
VP = tools_vanishing.detector_VP(folder_out)
# ---------------------------------------------------------------------------------------------------------------------

def example_01():
    image = cv2.imread('./images/ex_lines/chessboard10.jpg')
    p_src = numpy.array([(1659, 1029), (432, 103), (1380, 94), (138, 1052)], dtype=numpy.float32)

    H, W = image.shape[:2]
    target_width, target_height = int(H * 0.75), H
    h_ipersp = VP.get_four_point_transform_mat(p_src, target_width, target_height)
    image_wrapped = cv2.warpPerspective(image, h_ipersp, (target_width, target_height), borderValue=(32, 32, 32))
    cv2.imwrite(folder_out + 'wrapped1.png', image_wrapped)
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_02():
    image = cv2.imread('./images/ex_lines/frame000000.jpg')
    H, W = image.shape[:2]
    target_BEV_W, target_BEV_H = int(H * 0.75), H
    h_ipersp = VP.get_inverce_perspective_mat_v2(image, target_BEV_W, target_BEV_H, point_van_xy=(628, 88))
    image_wrapped = cv2.warpPerspective(image, h_ipersp, (target_BEV_W, target_BEV_H), borderValue=(32, 32, 32))
    cv2.imwrite(folder_out + 'wrapped2.png', image_wrapped)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_02()