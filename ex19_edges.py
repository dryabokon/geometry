
import numpy
import cv2
import tools_image
import warnings
import tools_draw_numpy
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------------------------
from CV import tools_Skeletone
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_lines/input250_img.jpg'
# ----------------------------------------------------------------------------------------------------------------------
Ske = tools_Skeletone.Skelenonizer(folder_out)
W,H = 3*255,255
# ----------------------------------------------------------------------------------------------------------------------
def ex_grid_search(filename_in):

    image = cv2.imread(filename_in)

    for th1 in range(0, 255, 10):
        for th2 in range(th1 + 1, 255, 10):
            image_edges = cv2.Canny(image=image, threshold1=th1, threshold2=th2)
            #image_edges = cv2.resize(image_edges, (W, H))
            image_edges = cv2.putText(image_edges, '{0} {1}'.format(th1, th2), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
            cv2.imwrite(folder_out + 'res_%03d_%03d.png'%(th1,th2), image_edges)

    cv2.imwrite(folder_out + 'res.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_00_edges_canny(filename_in):
    image = cv2.imread(filename_in)
    image_edges = cv2.Canny(image=image, apertureSize=3,threshold1=20, threshold2=150)
    cv2.imwrite(folder_out + 'canny.png', image_edges)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_00_edges_findContours(filename_in):

    image = cv2.imread(filename_in)
    binarized = Ske.binarize(image,maxValue=155)
    contours, hierarchy = cv2.findContours(binarized, 1, 2)
    # contours2, hierarchy2 = [], []
    # for c,h in zip(contours, hierarchy):
    #     if len(c) < 450:
    #         continue
    #     contours2.append(c)
    #     hierarchy2.append(h)

    image_cnt = numpy.zeros((image.shape[0], image.shape[1],3), dtype=numpy.uint8)
    contours = [c for c in contours if len(c) > 450]
    #image_cnt = cv2.drawContours(image_cnt, contours, -1, (255, 255, 255))
    colors = tools_draw_numpy.get_colors(len(contours))
    for c,color in zip(contours,colors):
        image_cnt = tools_draw_numpy.draw_contours(image_cnt, c.reshape((-1,2)), color=color,w=1,transperency=0.80)
    cv2.imwrite(folder_out + 'contours.png', image_cnt)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_01_skeletonize(filename_in):

    image = cv2.imread(filename_in)
    image_bin = Ske.binarize(Ske.preprocess_amplify(image))
    image_ske = Ske.binarized_to_skeleton(image_bin)
    segments = Ske.skeleton_to_segments(image_ske)
    segments_straight = Ske.sraighten_segments(segments, min_len=20)
    segments_long = Ske.filter_short_segments2(segments_straight,ratio=0.10)
    lines,losses = Ske.interpolate_segments_by_lines(segments_long)

    # cv2.imwrite(folder_out + 'filtered.png', result)
    # cv2.imwrite(folder_out + 'filtered2.png', result2)
    # cv2.imwrite(folder_out + 'ske.png', image_ske)
    cv2.imwrite(folder_out + 'segm_1_straight.png',tools_draw_numpy.draw_segments(tools_image.desaturate(image), segments_straight, colors, w=1))
    cv2.imwrite(folder_out + 'segm_2_long.png' ,tools_draw_numpy.draw_segments(tools_image.desaturate(image), segments_long, w=2))
    cv2.imwrite(folder_out + 'segm_3_lines.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, w=2))
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02_LSD(filename_in):

    image = cv2.imread(filename_in)
    lines = Ske.detect_lines_LSD(image)
    cv2.imwrite(folder_out + 'LSD_lines.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, w=2))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,'*.png')

    ex_grid_search(filename_in)

    # example_00_edges_canny(filename_in)
    # example_00_edges_findContours(filename_in)
    # example_01_skeletonize(filename_in)
    # example_02_LSD(filename_in)