#https://github.com/WolfgangFahl/play-chess-with-a-webcam/blob/master/examples/opencv/warp.py
import cv2
import numpy
import tools_Hough
import tools_Skeletone
import tools_render_CV
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
Ske = tools_Skeletone.Skelenonizer(folder_out)
Hough = tools_Hough.Hough()
# ---------------------------------------------------------------------------------------------------------------------

def example_01():
    image = cv2.imread('./images/ex_lines/chessboard10.jpg')
    #p_src = numpy.array([(432, 103), (1380, 94), (1659, 1029), (138, 1052)], dtype=numpy.float32)
    p_src = numpy.array([(1659, 1029), (432, 103), (1380, 94), (138, 1052)], dtype=numpy.float32)
    image_wrapped = tools_render_CV.four_point_transform(image, p_src)
    cv2.imwrite(folder_out + 'wrapped.png', image_wrapped)
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_02():
    image = cv2.imread('./images/ex_lines/frame000000.jpg')
    image_wrapped = tools_render_CV.inverce_perspective_mapping(image, (628, 88), (0, 275, 1280, 275))
    cv2.imwrite(folder_out + 'wrapped.png', image_wrapped)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_02()