import cv2
import os
# ---------------------------------------------------------------------------------------------------------------------
from CV import tools_alg_match
import tools_image
import tools_draw_numpy
import tools_IO


# ----------------------------------------------------------------------------------------------------------------------
def example_01_harris_corner_detection(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    for each in tools_alg_match.get_corners_Harris(img):
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
def example_02_shi_tomasi_corner_detector(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    for each in tools_alg_match.get_corners_Shi_Tomasi(img):
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
def example_03_SIFT(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    points, desc = tools_alg_match.get_keypoints_desc(img, detector='SIFT')

    for each in points:
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
def example_04_SURF(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    points, desc = tools_alg_match.get_keypoints_desc(img, detector='SURF')

    for each in points:
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
def example_05_Fast_Corner_Detector(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    points = tools_alg_match.get_corners_Fast(img)

    for each in points:
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
def example_06_ORB(filename_in, filename_out, R=2):
    img = cv2.imread(filename_in)
    gray_rgb = tools_image.desaturate(img)

    points, desc = tools_alg_match.get_keypoints_desc(img, detector='ORB')

    for each in points:
        gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], R, [0, 0, 255])

    cv2.imwrite(filename_out, gray_rgb)
    return


# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_keypoints/rght.jpg'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    else:
        tools_IO.remove_files(folder_out)

    example_01_harris_corner_detection(filename_in, folder_out + 'harris.png')
    example_02_shi_tomasi_corner_detector(filename_in, folder_out + 'shi.png')
    example_03_SIFT(filename_in, folder_out + 'sift.png')
    #example_04_SURF(filename_in, path_out + 'surf.png')
    example_05_Fast_Corner_Detector(filename_in, folder_out + 'fast_corners.png')
    example_06_ORB(filename_in, folder_out + 'orb.png')
