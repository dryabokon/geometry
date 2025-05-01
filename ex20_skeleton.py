import numpy
import cv2
import tools_image
import tools_draw_numpy
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
from CV import tools_Skeletone
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_lines/image1.jpg'
# ----------------------------------------------------------------------------------------------------------------------
S = tools_Skeletone.Skelenonizer(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out, '*.png')
    image = cv2.imread(filename_in)

    binarized = S.binarize(image,blockSize=27)
    cv2.imwrite(folder_out + 'bin.png', binarized)

    # edges = cv2.Canny(image=image, threshold1=0, threshold2=255)
    # cv2.imwrite(folder_out + 'edges.png', edges)

    # morphed = S.morph(binarized,kernel_h=3,kernel_w=3,n_dilate=3,n_erode=5)
    # cv2.imwrite(folder_out + 'edges_morphed.png', morphed)

    ske = S.skelenonize_fast(binarized)
    cv2.imwrite(folder_out + 'skelenonize_fast.png', ske)

    # segments = S.segmentize_slow(morphed)
    # cv2.imwrite(folder_out + 'skelenonize_sknw.png', tools_draw_numpy.draw_segments(binarized,segments))


