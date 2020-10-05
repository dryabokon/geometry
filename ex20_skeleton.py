import numpy
import cv2
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
import tools_Skeletone
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_ellipses/Image1_soccer.png'
#filename_in = './images/ex_lines/frame000159.jpg'
# ----------------------------------------------------------------------------------------------------------------------
def get_contours():
    #same as Canny
    _, contours, hierarchy = cv2.findContours(binarized, 1, 2)
    contours = [c for c in contours if len(c) > 450]
    image_cnt = numpy.zeros((image.shape[0], image.shape[1]), dtype=numpy.uint8)
    cv2.drawContours(image_cnt, contours, -1, (0, 0, 255))
    cv2.imwrite(folder_out + 'contours.png', edges)
    return
# ----------------------------------------------------------------------------------------------------------------------
S = tools_Skeletone.Skelenonizer()
if __name__ == '__main__':
    image = cv2.imread(filename_in)

    binarized = S.binarize(image)
    cv2.imwrite(folder_out + 'bin.png', binarized)

    edges = cv2.Canny(image=image, threshold1=0, threshold2=255)
    cv2.imwrite(folder_out + 'edges.png', edges)

    morphed = S.morph(edges,kernel_h=3,kernel_w=3,n_dilate=3,n_erode=0)
    cv2.imwrite(folder_out + 'edges_morphed.png', morphed)

    ske = S.binarized_to_skeleton_kiyko(morphed)
    cv2.imwrite(folder_out + 'ske_kiyko.png', ske)

    ske2 = S.binarized_to_skeleton_ski(morphed)
    cv2.imwrite(folder_out + 'ske_skimage.png', ske2)

