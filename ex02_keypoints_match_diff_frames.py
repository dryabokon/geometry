import cv2
import os
import numpy
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import math
# ---------------------------------------------------------------------------------------------------------------------
import tools_alg_match
import tools_image
import tools_draw_numpy
import tools_calibrate
import tools_IO


# ---------------------------------------------------------------------------------------------------------------------
def example_find_matches_for_frames(detector='SIFT', matchtype='knn'):

    folder_input = 'images/ex_frames/'
    img1 = cv2.imread(folder_input + '0045a.jpg')
    img2 = cv2.imread(folder_input + '0046a.jpg')

    #img1 = cv2.resize(img1, (1920,1080))
    #img2 = cv2.resize(img2, (1920,1080))

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)

    folder_output = 'images/output/'
    output_filename1 = folder_output + 'out1.png'
    output_filename2 = folder_output + 'out2.png'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)


    match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)


    for m1,m2 in zip(match1,match2):
        if math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) > 1 and \
                math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) < 55:
            color = [0, 0, 255]
            xxx = 127+255*math.atan(m1[1]-m2[1]/(m1[0]-m2[0]))/3.14
            xxx = (128+xxx)%256
            color = tools_image.hsv2bgr([xxx,255,255])
            img1_gray_rgb = tools_draw_numpy.draw_line(img1_gray_rgb, int(m1[1]), int(m1[0]), int(m2[1]), int(m2[0]), color)
            img2_gray_rgb = tools_draw_numpy.draw_line(img2_gray_rgb, int(m1[1]), int(m1[0]), int(m2[1]), int(m2[0]), color)


    '''
    for m1,m2 in zip(match1,match2):
        if math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) > 5 and \
                math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) < 55:
            r = int(255 * numpy.random.rand())
            color = cv2.cvtColor(numpy.array([r, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)

            img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(m1[1]), int(m1[0]), 4,color,alpha_transp=0.3)
            img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(m2[1]), int(m2[0]), 4,color,alpha_transp=0.3)
    '''

    cv2.imwrite(output_filename1, img1_gray_rgb)
    cv2.imwrite(output_filename2, img2_gray_rgb)

    return

# --------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_find_matches_for_frames('SIFT', 'flann')
