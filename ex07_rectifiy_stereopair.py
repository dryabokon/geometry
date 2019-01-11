import cv2
import os
import numpy
# --------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO


# --------------------------------------------------------------------------------------------------------------------
def example_rectify_pair():

    folder_calibr = 'images/ex06/'

    folder_input = 'images/ex06/'
    filename_in1 = '01.jpg'
    filename_in2 = '02.jpg'
    folder_output = 'images/output/'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    chess_rows, chess_cols = 7, 7

    im1 = cv2.imread(folder_input+filename_in1)
    im2 = cv2.imread(folder_input+filename_in2)

    camera_matrix, dist = tools_calibrate.get_proj_dist_mat_for_images(folder_calibr, chess_rows, chess_cols)

    #camera_matrix = numpy.array([[im1.shape[1], 0, im1.shape[0]], [0, im1.shape[0], im1.shape[1]], [0, 0, 1]]).astype(numpy.float64)
    #dist = numpy.zeros((1, 5))

    im1_remapped, im2_remapped = tools_calibrate.rectify_pair(camera_matrix, dist, im1, im2, chess_rows, chess_cols)
    cv2.imwrite(folder_output+filename_in1, im1_remapped)
    cv2.imwrite(folder_output+filename_in2, im2_remapped)

    return


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_rectify_pair()
