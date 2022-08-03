import cv2
import os
import numpy
from cv2 import aruco
# ---------------------------------------------------------------------------------------------------------------------
from CV import tools_calibrate
import tools_IO
import tools_image
import tools_draw_numpy
import tools_render_CV
from CV import tools_pr_geom
# ---------------------------------------------------------------------------------------------------------------------
def get_image_grid(W,H,dist = 0.0):

    image_grid = numpy.full((H, W, 3), 255, numpy.uint8)
    image_grid[numpy.arange(0, H, 10), :] = 128
    image_grid[:, numpy.arange(0, W, 10), :] = 128
    return image_grid
# ---------------------------------------------------------------------------------------------------------------------
def example_calibrate_camera_chess():

    folder_input  = 'images/ex_chessboard/'
    filename_input = '01.jpg'
    folder_output = 'images/output/'
    chess_rows,chess_cols  = 7, 7

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    camera_matrix, dist,rvecs, tvecs = tools_calibrate.get_proj_dist_mat_for_images(folder_input, chess_rows, chess_cols, folder_out=folder_output)

    image_chess = cv2.imread(folder_input+ filename_input)
    undistorted_chess = cv2.undistort(image_chess, camera_matrix, dist, None, None)
    cv2.imwrite(folder_output+'undistorted_'+filename_input, undistorted_chess)

    image_grid = get_image_grid(image_chess.shape[1],image_chess.shape[0])

    undistorted_grid = cv2.undistort(image_grid, camera_matrix, dist, None, None)
    cv2.imwrite(folder_output+'undistorted_grid.jpg', undistorted_grid)

    print(camera_matrix)

    return
# -------------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# -------------------------------------------------------------------------------------------------------------------------
def example_calibrate_aruco_markers(filename_in, marker_length_mm = 3.75, marker_space_mm = 0.5, dct = aruco.DICT_6X6_1000):

    image = cv2.imread(filename_in)
    gray = tools_image.desaturate(image)

    scale = (marker_length_mm / 2, marker_length_mm / 2, marker_length_mm / 2)
    num_cols, num_rows = 4,5
    board = aruco.GridBoard_create(num_cols, num_rows, marker_length_mm, marker_space_mm,aruco.getPredefinedDictionary(dct))
    image_AR, image_cube = gray.copy(), gray.copy()
    base_name, ext = filename_in.split('/')[-1].split('.')[0], filename_in.split('/')[-1].split('.')[1]
    #board_width_px = int((num_cols * marker_length_mm + (num_cols - 1) * marker_space_mm))
    #board_height_px= int((num_rows * marker_length_mm + (num_rows - 1) * marker_space_mm))
    #image_board = aruco.drawPlanarBoard(board, (board_width_px, board_height_px))


    camera_matrix = None#tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0])
    corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), aruco.getPredefinedDictionary(dct))

    if len(corners)>0:
        if len(corners)==1:corners,ids = numpy.array([corners]),numpy.array([ids])
        else:corners, ids = numpy.array(corners), numpy.array(ids)
        counters = numpy.array([len(ids)])

        ret, camera_matrix, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners,ids,counters, board, gray.shape[:2],None, None)
        image_markers = [tools_image.saturate(aruco.drawMarker(aruco.getPredefinedDictionary(dct), id, 100)) for id in ids]

        #!!!
        #camera_matrix = tools_pr_geom.compose_projection_mat_3x3(4200,4200)
        for i,image_marker in enumerate(image_markers):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_length_mm, camera_matrix, numpy.zeros(5))


            image_AR  = tools_render_CV.draw_image(image_AR,image_marker, camera_matrix, numpy.zeros(5), numpy.array(rvecs).flatten(), numpy.array(tvecs).flatten(),scale)
            image_cube = tools_render_CV.draw_cube_numpy(image_cube, camera_matrix, numpy.zeros(5), numpy.array(rvecs).flatten(),numpy.array(tvecs).flatten(), scale)

    cv2.imwrite(folder_out + base_name+'_AR.png', image_AR)
    cv2.imwrite(folder_out + base_name+'_AR_cube.png', image_cube)

    print(camera_matrix)

    return camera_matrix
# -------------------------------------------------------------------------------------------------------------------------
def example_calibrate_folder(folder_in, folder_out,marker_length_mm = 3.75, marker_space_mm = 0.5,dct = aruco.DICT_6X6_1000):
    tools_IO.remove_files(folder_out)

    num_cols, num_rows = 4, 5
    board = aruco.GridBoard_create(num_cols, num_rows, marker_length_mm, marker_space_mm,aruco.getPredefinedDictionary(dct))
    filenames = numpy.unique(tools_IO.get_filenames(folder_in, '*.jpg,*.png'))[:3]

    counter, corners_list, id_list, first = [], [], [], True
    for filename_in in filenames:
        base_name, ext = filename_in.split('/')[-1].split('.')[0], filename_in.split('/')[-1].split('.')[1]
        image = cv2.imread(folder_in + filename_in)
        img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(dct))
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = numpy.vstack((corners_list, corners))
            id_list = numpy.vstack((id_list,ids))
        counter.append(len(ids))

        image_temp = tools_image.desaturate(image.copy())
        aruco.drawDetectedMarkers(image_temp, corners)
        cv2.imwrite(folder_out+base_name+'.png',image_temp)
        print(base_name)


    counter = numpy.array(counter)
    ret, camera_matrix, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print(camera_matrix)

    return
# -------------------------------------------------------------------------------------------------------------------------
def load_points(filename_in):
    X = tools_IO.load_mat_pd(filename_in)
    IDs = numpy.array(X[:, 0],dtype=numpy.int32)
    X = X[:, 1:]
    points_2d = numpy.array(X[:, :2], dtype=numpy.float32)
    points_gps = numpy.array(X[:, 2:], dtype=numpy.float32)
    return points_2d, points_gps, IDs
# ----------------------------------------------------------------------------------------------------------------------
def shift_origin(points_gnss_3d, origin_xy, orignin_z=0, scale_factor_xy=100000.0):
    points_gnss_3d_normed = points_gnss_3d.copy()


    points_gnss_3d_normed[:,[0,1]]-=origin_xy[[0, 1]]
    points_gnss_3d_normed[:,[0,1]]*=scale_factor_xy
    points_gnss_3d_normed[:,2]=orignin_z
    return points_gnss_3d_normed
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_K_bruteforce_F(filename_image, filename_points, f_min=1920, f_max=10000):
    base_name = filename_image.split('/')[-1].split('.')[0]
    image = cv2.imread(filename_image)
    gray = tools_image.desaturate(image)
    points_2d_all, points_gps_all,IDs = load_points(filename_points)

    for f in numpy.arange(f_min,f_max,(f_max-f_min)/100):
        points_xyz = shift_origin(points_gps_all, points_gps_all[0])
        points_xyz, points_2d = points_xyz[1:], points_2d_all[1:]
        labels = ['(%2.1f,%2.1f)' % (p[0], p[1]) for p in points_xyz]

        image_AR = gray.copy()
        camera_matrix = numpy.array([[f, 0., 1920 / 2], [0., f, 1080 / 2], [0., 0., 1.]])
        rvecs, tvecs, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))
        rvecs, tvecs = numpy.array(rvecs).flatten(), numpy.array(tvecs).flatten()

        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d, color=(0, 0, 190), w=16,labels=labels)
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=8)
        for R in [10,100,1000]:image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs, tvecs, R)

        cv2.imwrite(folder_out + base_name + '_%05d'%(f) + '.png', image_AR)

    return
# -------------------------------------------------------------------------------------------------------------------------
def ex_undistort():
    # W, H = 1200,800
    # dist = 0.1
    # image_grid = get_image_grid(W,H)


    image_grid = cv2.imread('./images/ex_BEV/CityHealth/00000.jpg')
    H,W = image_grid.shape[:2]
    camera_matrix = numpy.array([[W, 0., W / 2], [0., H, H / 2], [0., 0., 1.]])
    dist = -0.85

    undistorted_grid = cv2.undistort(image_grid, camera_matrix, dist, None, None)
    cv2.imwrite(folder_out + 'undistorted_grid.jpg', undistorted_grid)
    return
# -------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #tools_IO.remove_files(folder_out)
    ex_undistort()
    #example_calibrate_camera_chess()
    #example_calibrate_aruco_markers('./images/ex_aruco/01.jpg', marker_length_mm=100 , marker_space_mm=5, dct=aruco.DICT_6X6_50)
    #example_calibrate_folder('./images/ex_aruco/cam01/',folder_out,dct=aruco.DICT_4X4_50)
    #evaluate_K_bruteforce_F('./images/ex_calibration/01000.jpg', './images/ex_calibration/points.csv', f_min=1920, f_max=10000)




