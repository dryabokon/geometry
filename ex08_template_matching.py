import cv2
import numpy
import os
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from sklearn.preprocessing import scale
# ---------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO
import tools_alg_match
import tools_image
import tools_draw_numpy


# ---------------------------------------------------------------------------------------------------------------------
def example_00():

    folder_input = 'images/ex_match/'
    folder_output = 'images/output/'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    img_base = cv2.imread(folder_input + 'boxes.png',0)
    img_tmpl = cv2.imread(folder_input + 'box.png',0)

    img_base = cv2.imread(folder_input + 'beach.png',0)
    img_tmpl = cv2.imread(folder_input + 'waldo.png',0)

    conv2 = fftconvolve(img_base, img_tmpl, mode='valid')
    conv2 = 255*((conv2 - numpy.min(conv2)) / numpy.ptp(conv2))

    cv2.imwrite(folder_output + 'hitmap2.png', conv2)
    return


# ---------------------------------------------------------------------------------------------------------------------
def example_01():
    folder_input = 'images/ex_match/'
    folder_output = 'images/output/'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    #img_base = cv2.imread(folder_input + 'boxes.png')
    #img_tmpl = cv2.imread(folder_input + 'pattern2.png')

    img_base = cv2.imread(folder_input + 'beach.png',0)
    img_tmpl = cv2.imread(folder_input + 'waldo.png',0)

    hitmap_2d = tools_alg_match.calc_hit_field_basic(img_base, img_tmpl)
    cv2.imwrite(folder_output + 'hitmap_basic.png', hitmap_2d)
    cv2.imwrite(folder_output + 'hitmap_jet.png', tools_image.hitmap2d_to_jet(hitmap_2d))
    cv2.imwrite(folder_output + 'hitmap_viridis.png', tools_image.hitmap2d_to_viridis(hitmap_2d))

    hitmap_2d = tools_alg_match.calc_hit_field(img_base, img_tmpl)
    cv2.imwrite(folder_output + 'hitmap_advanced.png', hitmap_2d)
    cv2.imwrite(folder_output + 'hitmap_advanced_jet.png', tools_image.hitmap2d_to_jet(hitmap_2d))
    cv2.imwrite(folder_output + 'hitmap_advanced_viridis.png', tools_image.hitmap2d_to_viridis(hitmap_2d))

    return


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_00()
    example_01()
