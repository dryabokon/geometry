import numpy
import cv2
import os
# ---------------------------------------------------------------------------------------------------------------------
from CV import tools_alg_match
import tools_image
import tools_draw_numpy
import tools_IO


# ----------------------------------------------------------------------------------------------------------------------
def draw_points(image, points, filename_out):
    image_res = tools_image.desaturate(image)
    image_res = tools_draw_numpy.draw_points(image_res, points, color=(0, 0, 200), w=4)
    cv2.imwrite(filename_out, image_res)

    return


# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#filename_in = './images/ex_keypoints/left.jpg'
#filename_in = './images/ex_keypoints/rght.jpg'
filename_in = './images/ex_chessboard/01.jpg'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    else:
        tools_IO.remove_files(folder_out)

    image = cv2.imread(filename_in)


    points_harris_corner_detection = tools_alg_match.get_corners_Harris(image)

    points_Shi_Tomasi = tools_alg_match.get_corners_Shi_Tomasi(image)
    points_sift, desc_sift = tools_alg_match.get_keypoints_desc(image, detector='SIFT')
    points_orb, desc_orb = tools_alg_match.get_keypoints_desc(image, detector='ORB')
    points_fast = tools_alg_match.get_corners_Fast(image)

    draw_points(image, points_harris_corner_detection, folder_out + 'harris.png')
    draw_points(image, points_Shi_Tomasi, folder_out + 'Shi_Tomasi.png')
    draw_points(image, points_sift, folder_out + 'sift.png')
    draw_points(image, points_orb, folder_out + 'orb.png')
    draw_points(image, points_fast, folder_out + 'fast_corners.png')
