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
import tools_render_CV
# ---------------------------------------------------------------------------------------------------------------------
def example_board_pose_estimation(folder_input, folder_output, chess_rows, chess_cols,cameraMatrix, dist):

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    R = 1

    axis_3d_end   = numpy.array([[R, 0, 0], [0, R, 0], [0, 0, -R]],dtype = numpy.float32)
    axis_3d_start = numpy.array([[0, 0, 0]],dtype=numpy.float32)

    points_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1], [-1, -1, +1], [-1, +1, +1], [+1, +1, +1],[+1, -1, +1]], dtype=numpy.float32)
    points_3d[:, [0,1,2]] *= 0.5
    points_3d[:, [0,1,2]] += 0.5
    points_3d[:, [2]] *= -1

    points_3d[:, [0]] += 3
    points_3d[:, [1]] += 4

    for image_name in fnmatch.filter(listdir(folder_input), '*.jpg'):

        img = cv2.imread(folder_input + image_name)
        image_AR = tools_image.desaturate(img)
        rvecs, tvecs = tools_calibrate.pose_estimation_chessboard(img, chess_rows, chess_cols, cameraMatrix, dist)

        if rvecs.size!=0:

            axis_2d_end, jac   = cv2.projectPoints(axis_3d_end  , rvecs, tvecs, cameraMatrix, dist)
            axis_2d_start, jac = cv2.projectPoints(axis_3d_start, rvecs, tvecs, cameraMatrix, dist)
            cv2.line(image_AR, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[0,0,0],axis_2d_end[0,0,1]), (0,0,255), thickness=3)
            cv2.line(image_AR, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[1,0,0],axis_2d_end[1,0,1]), (0,255,0), thickness=3)
            cv2.line(image_AR, (axis_2d_start[0,0,0], axis_2d_start[0,0,1]),(axis_2d_end[2,0,0],axis_2d_end[2,0,1]), (255,0,0), thickness=3)
            #image_AR = tools_render_CV.draw_cube_numpy(image_AR, camera_matrix, numpy.zeros(4), rvecs.flatten(), tvecs.flatten(),(0.5, 0.5, 0.5),points_3d=points_3d)
            image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, dist, numpy.array(rvecs).flatten(),numpy.array(tvecs).flatten(), R*5)


        cv2.imwrite(folder_output + image_name, image_AR)

    return
# ---------------------------------------------------------------------------------------------------------------------
def example_augment_board_live(chess_rows, chess_cols,camera_matrix, dist):
    cap = cv2.VideoCapture(0)

    axis_3d_end   = numpy.array([[3, 0, 0], [0, 3, 0], [0, 0, -3]],dtype = numpy.float32)
    axis_3d_start = numpy.array([[0, 0, 0]],dtype=numpy.float32)


    while (True):
        ret, img = cap.read()
        rvecs, tvecs = tools_calibrate.pose_estimation_chessboard(img, chess_rows, chess_cols, camera_matrix, dist)
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

    tools_IO.remove_files(path_output)

    camera_matrix, dist,rvecs, tvecs = tools_calibrate.get_proj_dist_mat_for_images(path_input, chess_rows, chess_cols)
    #camera_matrix = numpy.array([[2587.98 ,   0. ,   563.24], [   0. ,  2567.97 , 477.19], [   0.    ,  0.    ,  1.  ]])
    #dist = numpy.zeros(5)
    example_board_pose_estimation(path_input, path_output, chess_rows, chess_cols,camera_matrix, dist)




