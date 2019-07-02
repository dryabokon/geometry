import cv2
import os
import numpy
# ---------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO


# ---------------------------------------------------------------------------------------------------------------------
def example_calibrate_camera():

    folder_input  = 'images/ex06a/'
    filename_input = '01.jpg'
    folder_output = 'images/output/'
    chess_rows,chess_cols  = 6, 6

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    camera_matrix, dist = tools_calibrate.get_proj_dist_mat_for_images(folder_input, chess_rows, chess_cols, folder_output)

    image_chess = cv2.imread(folder_input+ filename_input)
    undistorted_chess = cv2.undistort(image_chess, camera_matrix, dist, None, None)
    cv2.imwrite(folder_output+'undistorted_'+filename_input, undistorted_chess)

    image_grid = numpy.full(image_chess.shape,255,numpy.uint8)
    image_grid[numpy.arange(0, image_chess.shape[0], 10),:] = 128
    image_grid[:,numpy.arange(0, image_chess.shape[1], 10), :] = 128

    undistorted_grid = cv2.undistort(image_grid, camera_matrix, dist, None, None)
    cv2.imwrite(folder_output+'undistorted_grid.jpg', undistorted_grid)

    return


# -------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_calibrate_camera()