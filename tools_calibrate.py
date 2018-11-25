import cv2
import numpy
from os import listdir
from glob import glob
import fnmatch
import math
#----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
#----------------------------------------------------------------------------------------------------------------------
def get_proj_dist_mat_for_image(filename,chess_rows,chess_cols):

    x, y = numpy.meshgrid(range(chess_rows), range(chess_cols))
    world_points = numpy.hstack((x.reshape(chess_rows*chess_cols, 1), y.reshape(chess_rows*chess_cols, 1), numpy.zeros((chess_rows*chess_cols, 1)))).astype(numpy.float32)
    im = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    _3d_points = []
    _2d_points = []

    ret, corners = cv2.findChessboardCorners(im, (chess_rows, chess_cols))

    mtx=[]
    dist=[]
    if ret:
        corners = cv2.cornerSubPix(im, corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        _2d_points.append(corners)
        _3d_points.append(world_points)

        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1], im.shape[0]), None, None)
        cv_im_undistorted = cv2.undistort(im, cameraMatrix, distCoeffs)
        projectPoints = numpy.array(cv2.projectPoints(world_points, numpy.array(rvecs), numpy.array(tvecs), cameraMatrix, distCoeffs))[0]
        projectPoints = projectPoints.reshape(projectPoints.shape[0], projectPoints.shape[2])
        corners = numpy.array(corners).reshape(corners.shape[0], corners.shape[2])

    return cameraMatrix, distCoeffs,cv_im_undistorted
#----------------------------------------------------------------------------------------------------------------------
def get_proj_dist_mat_for_images(folder_in,chess_rows,chess_cols,folder_out=None):

    x, y = numpy.meshgrid(range(chess_rows), range(chess_cols))
    world_points = numpy.hstack((x.reshape(chess_rows*chess_cols, 1), y.reshape(chess_rows*chess_cols, 1), numpy.zeros((chess_rows*chess_cols, 1)))).astype(numpy.float32)

    _3d_points = []
    _2d_points = []


    for image_name in fnmatch.filter(listdir(folder_in), '*.jpg'):
        im = cv2.imread(folder_in+image_name)
        im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray_RGB = tools_image.desaturate(im)
        ret, corners = cv2.findChessboardCorners(im_gray, (chess_rows, chess_cols))

        if ret:
            corners = cv2.cornerSubPix(im_gray, corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            _2d_points.append(corners)
            _3d_points.append(world_points)
            corners = corners.reshape(-1, 2)
            for i in range(0,corners.shape[0]):
                im_gray_RGB = tools_draw_numpy.draw_circle(im_gray_RGB, corners[i, 1], corners[i, 0], 3, [0, 0, 255], alpha_transp=0)

        if folder_out!=None:
            cv2.imwrite(folder_out + image_name, im_gray_RGB)

    camera_matrix = numpy.array([[im.shape[1], 0, im.shape[0]], [0, im.shape[0], im.shape[1]], [0, 0, 1]]).astype(numpy.float64)
    dist=numpy.zeros((1,5))

    flag = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL

    matrix_init = numpy.zeros((3, 3), numpy.float32)
    matrix_init[0][0] = im.shape[0]/2
    matrix_init[0][2] = im.shape[1]/2
    matrix_init[1][1] = matrix_init[0][0]
    matrix_init[1][2] = matrix_init[0][2]
    matrix_init[2][2] = 1.0
    dist_init = numpy.zeros((1, 4), numpy.float32)
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1], im.shape[0]), matrix_init, dist_init,flags=flag)

    return camera_matrix,dist
#----------------------------------------------------------------------------------------------------------------------
def rectify_pair(mtx,dist,image1,image2,chess_rows,chess_cols):

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    x, y = numpy.meshgrid(range(chess_rows), range(chess_cols))
    world_points = numpy.hstack((x.reshape(chess_rows*chess_cols, 1), y.reshape(chess_rows*chess_cols, 1), numpy.zeros((chess_rows*chess_cols, 1)))).astype(numpy.float32)

    all_corners1 = []
    all_corners2 = []
    all_3d_points = []
    im1_remapped = []
    im2_remapped = []

    ret1, corners1 = cv2.findChessboardCorners(gray_image1, (chess_rows, chess_cols))
    ret2, corners2 = cv2.findChessboardCorners(gray_image2, (chess_rows, chess_cols))

    if ret1 and ret2:
        all_corners1.append(corners1)
        all_corners2.append(corners2)
        all_3d_points.append(world_points)

        flg = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_FIX_FOCAL_LENGTH +cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS
        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(all_3d_points, all_corners1, all_corners2,mtx, dist, mtx, dist,(gray_image1.shape[1], gray_image1.shape[0]),flags=flg)

        R1 = numpy.zeros((3, 3))
        R2 = numpy.zeros((3, 3))
        P1 = numpy.zeros((3, 4))
        P2 = numpy.zeros((3, 4))

        RL, RR, PL, PR, _, _, _ = cv2.stereoRectify(mtx,dist,mtx,dist,(gray_image1.shape[1],gray_image1.shape[0]),R,T, R1,R2,P1,P2)

        R1 = numpy.array(R1)
        R2 = numpy.array(R2)
        P1 = numpy.array(P1)
        P2 = numpy.array(P2)

        map1_x, map1_y = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (gray_image1.shape[1],gray_image1.shape[0]),cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (gray_image1.shape[1],gray_image1.shape[0]),cv2.CV_32FC1)

        im1_remapped = cv2.remap(image1, map1_x, map1_y, cv2.INTER_LINEAR)
        im2_remapped = cv2.remap(image2, map2_x, map2_y, cv2.INTER_LINEAR)

    return im1_remapped, im2_remapped
# ---------------------------------------------------------------------------------------------------------------------
def get_stitched_images_using_homography(img1, img2, M,background_color=(255, 255, 255)):
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims      = numpy.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = numpy.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)  # Get relative perspective of second image
    result_dims = numpy.concatenate((img1_dims, img2_dims), axis=0)  # Resulting dimensions

    [x_min, y_min] = numpy.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = numpy.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])

    # Warp images to get the resulting image
    result_img_2 = cv2.warpPerspective(img2, transform_array.dot(M), (x_max - x_min, y_max - y_min),borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    result_img_1 = numpy.full(result_img_2.shape,background_color,dtype=numpy.uint8)
    result_img_1[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]] = img1

    return result_img_1, result_img_2
# ---------------------------------------------------------------------------------------------------------------------
def get_stitched_images_using_translation(img1, img2, M,background=(255, 255, 255)):
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]


    img1_dims      = numpy.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = numpy.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    img2_dims = cv2.transform(img2_dims_temp, M)  # Get relative perspective of second image
    result_dims = numpy.concatenate((img1_dims, img2_dims), axis=0)  # Resulting dimensions

    [x_min, y_min] = numpy.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]

    M[0, 2]+=-x_min
    M[1, 2]+=-y_min

    result_img1 = cv2.warpAffine(img2, M, (x_max - x_min, y_max - y_min),borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    result_img2 = numpy.full(result_img1.shape,background)
    result_img2 [transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]] = img1

    return result_img1,result_img2
# ---------------------------------------------------------------------------------------------------------------------
def get_stitched_images_middle(img1,img2,H1,H2,background=(255, 255, 255)):
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1_dims = cv2.perspectiveTransform(numpy.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2), H1)
    img2_dims = cv2.perspectiveTransform(numpy.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2), H2)

    result_dims = numpy.concatenate((img1_dims, img2_dims), axis=0)
    [x_min, y_min] = numpy.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]
    transform_array = numpy.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])

    result_img1 = cv2.warpPerspective(img1, transform_array.dot(H1), (x_max - x_min, y_max - y_min),borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    result_img2 = cv2.warpPerspective(img2, transform_array.dot(H2), (x_max - x_min, y_max - y_min),borderMode=cv2.BORDER_CONSTANT, borderValue=background)

    return result_img1,result_img2
# --------------------------------------------------------------------------------------------------------------------------
def get_matches_on_desc(des_source, des_destin):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_destin, des_source)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
# --------------------------------------------------------------------------------------------------------------------------
def get_matches_on_desc_knn(des_source, des_destin):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_destin, des_source, k=2)

    verify_ratio = 0.8
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    return verified_matches
# --------------------------------------------------------------------------------------------------------------------------
def get_matches_on_desc_flann(des_source, des_destin):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_destin.astype(numpy.float32), des_source.astype(numpy.float32), k=2)

    verify_ratio = 0.8
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    return verified_matches
# --------------------------------------------------------------------------------------------------------------------------
def get_homography_by_keypoints_desc(points_source,des_source, points_destin,des_destin,matchtype='knn'):

    src, dst, distance = get_matches_from_keypoints_desc(points_source, des_source, points_destin, des_destin, matchtype='knn')
    M = []
    if src.size!=0:
        M = get_homography_by_keypoints(src, dst)
    return M
# ---------------------------------------------------------------------------------------------------------------------
def get_matches_from_keypoints_desc(points_source,des_source, points_destin,des_destin,matchtype='knn'):

    if matchtype=='knn':
        matches = get_matches_on_desc_knn(des_source, des_destin)
    elif matchtype=='flann':
        matches = get_matches_on_desc_flann(des_source, des_destin)
    else:
        if des_destin.shape[0]>des_source.shape[0]:
            zzz = numpy.zeros((+des_destin.shape[0] - des_source.shape[0], des_destin.shape[1]))
            des_source=numpy.vstack((des_source,zzz))
        if des_destin.shape[0]<des_source.shape[0]:
            zzz = numpy.zeros((-des_destin.shape[0] + des_source.shape[0], des_destin.shape[1]))
            des_destin=numpy.vstack((des_destin,zzz))

        matches = get_matches_on_desc(des_source.astype(numpy.uint8),des_destin.astype(numpy.uint8) )

    src,dst, distance= [],[],[]
    for m in matches:
        if m.queryIdx < points_destin.shape[0] and m.trainIdx < points_source.shape[0]:
            src.append(points_source[m.trainIdx])
            dst.append(points_destin[m.queryIdx])
            distance.append(m.distance)

    return numpy.array(src), numpy.array(dst), numpy.array(distance)
# ---------------------------------------------------------------------------------------------------------------------
def get_matches_from_desc_limit_by_disp(points_source,des_source, points_destin,des_destin,disp_v1, disp_v2, disp_h1, disp_h2,matchtype='knn'):

    all_matches_scr = numpy.zeros((1,2))
    all_matches_dst = numpy.zeros((1, 2))
    distances = []
    for i in range (0,len(points_source)):
        row1, col1 = points_source[i][1], points_source[i][0]
        idx = []
        for j in range(0, len(points_destin)):
            row2, col2 = points_destin[j][1], points_destin[j][0]
            if (col2 - col1 >= disp_h1) and (col2 - col1 < disp_h2) and (row2 - row1 >= disp_v1) and (row2 - row1 < disp_v2):
                idx.append(j)

        idx = numpy.array(idx)

        if idx.size!=0:
            pnts_src = numpy.array(points_source[i]).reshape(-1, 2)
            des_src  = numpy.array(des_source[i]).reshape(-1, 2)
            pnts_dest = numpy.array(points_destin[idx]).reshape(-1, 2)
            des_dest = numpy.array(des_destin[idx]).reshape(-1, 2)
            match1, match2, dist = get_matches_from_keypoints_desc(pnts_src,des_src , pnts_dest, des_dest, matchtype)
            if match1.size!=0:
                all_matches_scr = numpy.vstack((all_matches_scr, match1))
                all_matches_dst = numpy.vstack((all_matches_dst, match2))
                distances.append(dist)

    all_matches_scr = numpy.array(all_matches_scr).astype(numpy.int)
    all_matches_dst = numpy.array(all_matches_dst).astype(numpy.int)


    return all_matches_scr[1:], all_matches_dst[1:], distances
# ---------------------------------------------------------------------------------------------------------------------
def get_homography_by_keypoints(src,dst):
    method = cv2.RANSAC
    #method = cv2.LMEDS
    #method = cv2.RHO
    M, mask = cv2.findHomography(src, dst, method, 3.0)
    return M
#----------------------------------------------------------------------------------------------------------------------
def get_homography_middle(img_source, img_destination):

    sift = cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(img_source, None)
    k2, d2 = sift.detectAndCompute(img_destination, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    verify_ratio = 0.8  # Source: stackoverflow
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    img1_pts = []
    img2_pts = []

    for match in verified_matches:
        img1_pts.append(k1[match.queryIdx].pt)
        img2_pts.append(k2[match.trainIdx].pt)
    img1_pts = numpy.float32(img1_pts).reshape(-1, 1, 2)
    img2_pts = numpy.float32(img2_pts).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

    H = M
    H_inv = get_inverse_homography(M)

    img2_pts_transformed_all = []
    img1_pts_transformed_all = []
    for alpha in range(0,100):
        HB1 = bend_homography(H    , alpha/99.0)
        HB2 = bend_homography(H_inv, alpha/99.0)
        img1_pts_transformed_all.append(cv2.perspectiveTransform(img1_pts, HB1).reshape(img1_pts.shape[0],2))
        img2_pts_transformed_all.append(cv2.perspectiveTransform(img2_pts, HB2).reshape(img1_pts.shape[0],2))

    th = 2

    Hits_large = numpy.zeros((100,100))

    for i in range(0,100):
        for j in range(0, 100):
            dist = numpy.abs(img1_pts_transformed_all[i] - img2_pts_transformed_all[j])
            error = numpy.sum(dist,axis=1)
            Hits_large[i,j] = numpy.sum(1*(error<=th))

    Hits = Hits_large[10:90,10:90]
    i=numpy.argmax(Hits)
    alpha1 = int(i / Hits.shape[1])+10
    alpha2 = int(i % Hits.shape[1])+10

    HB1 = bend_homography(H    , alpha1 / 99.0)
    HB2 = bend_homography(H_inv, alpha2 / 99.0)
    img1_pts_transformed_all = cv2.perspectiveTransform(img1_pts, HB1).reshape(img1_pts.shape[0], 2)
    img2_pts_transformed_all = cv2.perspectiveTransform(img2_pts, HB2).reshape(img1_pts.shape[0], 2)

    #print(img1_pts_transformed_all-img2_pts_transformed_all)

    return HB1,HB2
#----------------------------------------------------------------------------------------------------------------------
def rotationMatrixToEulerAngles(R):

    Rt = numpy.transpose(R)
    shouldBeIdentity = numpy.dot(Rt, R)
    I = numpy.identity(3, dtype=R.dtype)
    n = numpy.linalg.norm(I - shouldBeIdentity)

    if(n < 1e-6):

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

    return numpy.array([x, y, z])
#----------------------------------------------------------------------------------------------------------------------
def eulerAnglesToRotationMatrix(theta):

    R_x = numpy.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = numpy.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = numpy.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = numpy.dot(R_z, numpy.dot(R_y, R_x))

    return R
#----------------------------------------------------------------------------------------------------------------------
def derive_transform(img1,img2,K=numpy.array([[1000,0,0],[0,1000,0],[0,0,1]])):

    H = get_sift_homography(img1, img2).astype('float')
    n, R, T, normal = cv2.decomposeHomographyMat(H, K)
    R = numpy.array(R[0])
    T = numpy.array(T[0])
    normal = numpy.array(normal[0])

    HH=compose_homography(R,T,normal,K)

    return R,T,normal,HH,K
#----------------------------------------------------------------------------------------------------------------------
def compose_homography(R,T,normal,K=numpy.array([[1000,0,0],[0,1000,0],[0,0,1]])):

    HH= R + numpy.dot(T,normal.T)
    HH= numpy.dot(numpy.dot(K,HH),numpy.linalg.inv(K))
    HH/=HH[2,2]

    return HH
#----------------------------------------------------------------------------------------------------------------------
def get_inverse_homography_from_RT(R, T, normal,K=numpy.array([[1000,0,0],[0,1000,0],[0,0,1]])):
    HH= numpy.linalg.inv(R + numpy.dot(T,normal.T))
    HH= numpy.dot(numpy.dot(K,HH),numpy.linalg.inv(K))
    HH/=HH[2,2]
    return HH
#----------------------------------------------------------------------------------------------------------------------
def get_inverse_homography(H):
    v,HH = cv2.invert(H)
    HH /= HH[2, 2]
    return HH
#----------------------------------------------------------------------------------------------------------------------
def bend_homography(H,alpha,K=numpy.array([[1000,0,0],[0,1000,0],[0,0,1]])):
    n, R, T, normal = cv2.decomposeHomographyMat(H, K)

    R= R[0]
    T= T[0]
    normal = normal[0]

    angles = rotationMatrixToEulerAngles(R)*alpha
    R = eulerAnglesToRotationMatrix(angles)
    T = T*alpha

    H = compose_homography(R, T, normal, K)

    return H
# ---------------------------------------------------------------------------------------------------------------------
def align_images(im1, im2,mode = cv2.MOTION_AFFINE):

    if len(im1.shape) == 2:
        im1_gray = im1.copy()
        im2_gray = im2.copy()

    else:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    #mode = cv2.MOTION_TRANSLATION
    #mode = cv2.MOTION_AFFINE

    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(2, 3, dtype=numpy.float32),mode)
    except:
        return im1, im2

    if len(im1.shape)==2:
        aligned = cv2.warpAffine(im2_gray, warp_matrix, (im2_gray.shape[1], im2_gray.shape[0]),borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return im1_gray, aligned
    else:
        aligned = cv2.warpAffine(im2, warp_matrix, (im2_gray.shape[1], im2_gray.shape[0]),borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return im1, aligned
# ---------------------------------------------------------------------------------------------------------------------
def align_and_average(pattern,images):
    if(images.size==0):
        return pattern

    average = numpy.zeros((images.shape[1], images.shape[2],3))
    for i in range(0, images.shape[0]):
        pattern, aligned = align_images(pattern, images[i],mode = cv2.MOTION_AFFINE)
        average+=aligned

    return ((average)/images.shape[0]).astype(numpy.uint8)
# ---------------------------------------------------------------------------------------------------------------------
def get_rvecs_tvecs(img,chess_rows, chess_cols,cameraMatrix, dist):
    corners_3d = numpy.zeros((chess_rows * chess_cols, 3), numpy.float32)
    corners_3d[:, :2] = numpy.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners_2d = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)

    rvecs, tvecs=numpy.array([]),numpy.array([])

    if ret == True:
        corners_2d = cv2.cornerSubPix(gray, corners_2d, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(corners_3d, corners_2d, cameraMatrix, dist)

    return rvecs, tvecs
# ---------------------------------------------------------------------------------------------------------------------