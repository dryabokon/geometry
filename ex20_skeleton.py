import numpy
import cv2
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
from CV import tools_Skeletone
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_keypoints/_001644.jpg'
# ----------------------------------------------------------------------------------------------------------------------
S = tools_Skeletone.Skelenonizer(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    image = cv2.imread(filename_in)

    binarized = S.binarize(image)
    cv2.imwrite(folder_out + 'bin.png', binarized)

    edges = cv2.Canny(image=image, threshold1=0, threshold2=255)
    cv2.imwrite(folder_out + 'edges.png', edges)

    morphed = S.morph(edges,kernel_h=3,kernel_w=3,n_dilate=3,n_erode=0)
    cv2.imwrite(folder_out + 'edges_morphed.png', morphed)

    ske = S.skelenonize_fast(morphed)
    cv2.imwrite(folder_out + 'skelenonize_fast.png', ske)

    # segments = S.segmentize_slow(morphed)
    # cv2.imwrite(folder_out + 'skelenonize_sknw.png', tools_draw_numpy.draw_segments(binarized,segments))


