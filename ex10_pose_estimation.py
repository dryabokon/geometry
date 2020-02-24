import cv2
import numpy
from os import listdir
import fnmatch
import os
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO
import tools_alg_match
import tools_image
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------
def example_board_pose_estimation(folder_input, folder_output, chess_rows, chess_cols,cameraMatrix, dist):

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    R = 2

    axis_3d_end   = numpy.array([[R, 0, 0], [0, R, 0], [0, 0, -R]],dtype = numpy.float32)
    axis_3d_start = numpy.array([[0, 0, 0]],dtype=numpy.float32)

    for image_name in fnmatch.filter(listdir(folder_input), '*.jpg'):

        img = cv2.imread(folder_input + image_name)
        img_gray_rgb = tools_image.desaturate(img)
        rvecs, tvecs = tools_calibrate.get_rvecs_tvecs_for_chessboard(img, chess_rows, chess_cols, cameraMatrix, dist)

        if rvecs.size!=0:

            axis_2d_end, jac   = cv2.projectPoints(axis_3d_end  , rvecs, tvecs, cameraMatrix, dist)
            axis_2d_start, jac = cv2.projectPoints(axis_3d_start, rvecs, tvecs, cameraMatrix, dist)
            cv2.line(img_gray_rgb, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[0,0,0],axis_2d_end[0,0,1]), (0,0,255), thickness=3)
            cv2.line(img_gray_rgb, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[1,0,0],axis_2d_end[1,0,1]), (0,255,0), thickness=3)
            cv2.line(img_gray_rgb, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[2,0,0],axis_2d_end[2,0,1]), (255,0,0), thickness=3)

        cv2.imwrite(folder_output + image_name, img_gray_rgb)

    return
# ---------------------------------------------------------------------------------------------------------------------
def example_augment_board_live(chess_rows, chess_cols,camera_matrix, dist):
    cap = cv2.VideoCapture(0)

    axis_3d_end   = numpy.array([[3, 0, 0], [0, 3, 0], [0, 0, -3]],dtype = numpy.float32)
    axis_3d_start = numpy.array([[0, 0, 0]],dtype=numpy.float32)


    while (True):
        ret, img = cap.read()
        rvecs, tvecs = tools_calibrate.get_rvecs_tvecs_for_chessboard(img, chess_rows, chess_cols, camera_matrix, dist)
        if rvecs.size!=0:
            axis_2d_end, jac   = cv2.projectPoints(axis_3d_end  , rvecs, tvecs, camera_matrix, dist)
            axis_2d_start, jac = cv2.projectPoints(axis_3d_start, rvecs, tvecs, camera_matrix, dist)
            img = tools_draw_numpy.draw_line(img, axis_2d_start[0,0,1], axis_2d_start[0,0,0], axis_2d_end[0,0,1],axis_2d_end[0,0,0], (0,0,255))
            img = tools_draw_numpy.draw_line(img, axis_2d_start[0,0,1], axis_2d_start[0,0,0], axis_2d_end[1,0,1],axis_2d_end[1,0,0], (0,0,255))
            img = tools_draw_numpy.draw_line(img, axis_2d_start[0,0,1], axis_2d_start[0,0,0], axis_2d_end[2,0,1],axis_2d_end[2,0,0], (0,0,255))

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_input = 'images/ex_chessboard/'
    path_output = 'images/output/'
    chess_rows, chess_cols = 7,7
    width,height =1200, 800

    camera_matrix = numpy.array([[height,0,width],[0,width,height],[0,0,1]]).astype(numpy.float64)
    dist=numpy.zeros((1,5))
    camera_matrix, dist = tools_calibrate.get_proj_dist_mat_for_images(path_input, chess_rows, chess_cols)

    example_board_pose_estimation(path_input, path_output, chess_rows, chess_cols,camera_matrix, dist)
    #example_augment_board_live(8, 8,camera_matrix, dist)
