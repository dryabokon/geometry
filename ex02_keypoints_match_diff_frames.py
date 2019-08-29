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
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
def example_find_matches_for_frames(detector='SIFT', matchtype='knn'):

    folder_input = 'images/ex_homography_affine/'
    folder_output = 'images/output/'
    img1 = cv2.imread(folder_input + 'first.jpg')
    img2 = cv2.imread(folder_input + 'scond.jpg')

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)
    match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)
    for m1,m2 in zip(match1,match2):
        if math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) > 1 and math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) < 55:
            img1_gray_rgb = tools_draw_numpy.draw_line(img1_gray_rgb, int(m1[1]), int(m1[0]), int(m2[1]), int(m2[0]), [0, 0, 255])
            img2_gray_rgb = tools_draw_numpy.draw_line(img2_gray_rgb, int(m1[1]), int(m1[0]), int(m2[1]), int(m2[0]), [0, 0, 255])
    cv2.imwrite(folder_output + 'out1.png', img1_gray_rgb)
    cv2.imwrite(folder_output + 'out2.png', img2_gray_rgb)

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)
    for m1,m2 in zip(match1,match2):
        if math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) > 5 and \
                math.sqrt((m1[1]-m2[1])*(m1[1]-m2[1]) + (m1[0]-m2[0])*(m1[0]-m2[0])) < 55:
            r = int(255 * numpy.random.rand())
            color = cv2.cvtColor(numpy.array([r, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)

            img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(m1[1]), int(m1[0]), 4,color,alpha_transp=0.3)
            img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(m2[1]), int(m2[0]), 4,color,alpha_transp=0.3)
    cv2.imwrite(folder_output + 'out3.png', img1_gray_rgb)
    cv2.imwrite(folder_output + 'out4.png', img2_gray_rgb)


    return

# --------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    example_find_matches_for_frames('SIFT', 'flann')
