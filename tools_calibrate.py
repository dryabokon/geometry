import cv2
import numpy
from os import listdir
from glob import glob
import fnmatch
import math
#----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_alg_match
import tools_IO
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
def get_stitched_images_using_homography(img1, img2, M,background_color=(255, 255, 255),borderMode=cv2.BORDER_CONSTANT):

    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img2_dims      = numpy.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
    img1_dims_temp = numpy.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img1_dims = cv2.perspectiveTransform(img1_dims_temp, M)  # Get relative perspective of second image
    result_dims = numpy.concatenate((img1_dims, img2_dims), axis=0)  # Resulting dimensions

    [x_min, y_min] = numpy.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]
    transform_array = numpy.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])

    result_img_1 = cv2.warpPerspective(img1, transform_array.dot(M), (x_max - x_min, y_max - y_min),borderMode=borderMode, borderValue=background_color)

    result_img_2 = numpy.full(result_img_1.shape,background_color,dtype=numpy.uint8)
    result_img_2[transform_dist[1]:w2 + transform_dist[1], transform_dist[0]:h2 + transform_dist[0]] = img2
    if borderMode == cv2.BORDER_REPLICATE:
        result_img_2 = tools_image.fill_border(result_img_2,transform_dist[1],transform_dist[0],w2+transform_dist[1],h2 + transform_dist[0])

    return result_img_1, result_img_2
# ---------------------------------------------------------------------------------------------------------------------
def get_stitched_images_using_translation(img1, img2, translation,background_color=(255, 255, 255),borderMode=cv2.BORDER_CONSTANT,keep_shape=False):
    #cv2.BORDER_CONSTANT
    #cv2.BORDER_REPLICATE

    M = translation.copy()

    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img2_dims      = numpy.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
    img1_dims_temp = numpy.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)

    img1_dims = cv2.transform(img1_dims_temp, M)  # Get relative perspective of second image
    result_dims = numpy.concatenate((img1_dims, img2_dims), axis=0)  # Resulting dimensions

    [x_min, y_min] = numpy.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]

    M[0, 2]+=-x_min
    M[1, 2]+=-y_min

    result_img1 = cv2.warpAffine(img1, M, (x_max - x_min, y_max - y_min),borderMode=borderMode, borderValue=background_color)
    #print(M)
    #print(result_img1.shape)

    #!!!
    result_img2 = numpy.full(result_img1.shape,background_color,dtype=numpy.uint8)
    result_img2 [transform_dist[1]:w2 + transform_dist[1], transform_dist[0]:h2 + transform_dist[0]] = img2

    if keep_shape==False:
        if borderMode == cv2.BORDER_REPLICATE:
            result_img2 = tools_image.fill_border(result_img2,transform_dist[1],transform_dist[0],w2+transform_dist[1],h2 + transform_dist[0])
    else:
        result_img2 = result_img2[transform_dist[1]:transform_dist[1] + img1.shape[0],transform_dist[0]:transform_dist[0] + img1.shape[1]]
        result_img1 = result_img1[transform_dist[1]:transform_dist[1] + img1.shape[0],transform_dist[0]:transform_dist[0] + img1.shape[1]]

    return result_img1, result_img2

# --------------------------------------------------------------------------------------------------------------------------
def get_transform_by_keypoints_desc(points_source,des_source, points_destin,des_destin,matchtype='knn'):

    M = None
    if points_source==[] or des_source==[] or points_destin==[] or des_destin==[]:
        return M

    src, dst, distance = tools_alg_match.get_matches_from_keypoints_desc(points_source, des_source, points_destin, des_destin, matchtype=matchtype)

    if src.size!=0:
        M = get_transform_by_keypoints(src, dst)


    return M
# --------------------------------------------------------------------------------------------------------------------------
def get_homography_by_keypoints_desc(points_source,des_source, points_destin,des_destin,matchtype='knn'):

    M = None
    if points_source==[] or des_source==[] or points_destin==[] or des_destin==[]:
        return M

    src, dst, distance = tools_alg_match.get_matches_from_keypoints_desc(points_source, des_source, points_destin, des_destin, matchtype=matchtype)

    if src.size!=0:
        M = get_homography_by_keypoints(src, dst)
    return M
# ---------------------------------------------------------------------------------------------------------------------
def get_transform_by_keypoints(src,dst):

    #M,_ = cv2.estimateAffine2D(src, dst,confidence=0.95)
    M, _ = cv2.estimateAffinePartial2D(src, dst)

    return M
#----------------------------------------------------------------------------------------------------------------------


def get_homography_by_keypoints(src,dst):
    method = cv2.RANSAC
    #method = cv2.LMEDS
    #method = cv2.RHO
    M, mask = cv2.findHomography(src, dst, method, 3.0)

    return M

#----------------------------------------------------------------------------------------------------------------------
def rotationMatrixToEulerAngles(R):

    Rt = numpy.transpose(R)
    shouldBeIdentity = numpy.dot(Rt, R)
    I = numpy.identity(3, dtype=R.dtype)
    n = numpy.linalg.norm(I - shouldBeIdentity)

    if True or (n < 1e-6):

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
def align_two_images_translation(img1, img2,detector='SIFT',matchtype='knn',borderMode=cv2.BORDER_REPLICATE, background_color=(0, 255, 255)):

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)
    coord1, coord2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)

    translation= None
    if coord1 is not None and coord1.size >= 8:
        translation= get_transform_by_keypoints(coord1, coord2)
        if translation is None or math.isnan(translation[0,0]):
            return img1,img2,0
        #T = numpy.eye(3,dtype=numpy.float32)
        #T[0:2,0:2] = translation[0:2,0:2]
        #angle = math.fabs(rotationMatrixToEulerAngles(T)[2])

        #if translation[0, 2] >= 0.10*img1.shape[0] or translation[1, 2] >= 0.10*img1.shape[1]:# or angle>0.05:
        #    return img1, img2, 0
    else:
        return img1, img2,0

    if borderMode==cv2.BORDER_REPLICATE:
        result_image1, result_image2 = get_stitched_images_using_translation(img1, img2, translation, borderMode=cv2.BORDER_REPLICATE,keep_shape=True)
    else:

        result_image1a, result_image2 = get_stitched_images_using_translation(img1, img2, translation,borderMode=cv2.BORDER_CONSTANT,background_color=(0  ,255  ,0  ), keep_shape=True)
        result_image1b, result_image2 = get_stitched_images_using_translation(img1, img2, translation,borderMode=cv2.BORDER_CONSTANT,background_color=(255,0,255), keep_shape=True)

        result_image1 = result_image1b.copy()
        idx2 = numpy.all(result_image1a != result_image1b, axis=-1)
        result_image1[idx2] = background_color


    q = ((cv2.matchTemplate(result_image1,result_image2, method=cv2.TM_CCOEFF_NORMED)[0, 0])+1)*128

    return result_image1,result_image2,int(q)

# ---------------------------------------------------------------------------------------------------------------------
def align_two_images_homography(img1, img2,detector='SIFT',matchtype='knn'):

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)

    match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)

    homography= None
    if match1.size != 0:
        homography= get_homography_by_keypoints_desc(points1, des1, points2, des2, matchtype)
        if homography is None:
            return img1,img2
    else:
        return img1, img2


    for each in match1:
        cv2.circle(img1_gray_rgb, (int(each[0]), int(each[1])), 3, [0, 0, 255],thickness=-1)
    for each in match2:
        cv2.circle(img2_gray_rgb, (int(each[0]), int(each[1])), 3, [255, 255, 0], thickness=-1)


    result_image1, result_image2 = get_stitched_images_using_homography(img2_gray_rgb, img1_gray_rgb, homography, borderMode=cv2.BORDER_REPLICATE,background_color=(255, 255, 255))
    q = cv2.matchTemplate(result_image1, result_image2, method=cv2.TM_CCOEFF_NORMED)[0, 0]
    q = int((1 + q) * 128)

    return result_image1,result_image2, q

# ---------------------------------------------------------------------------------------------------------------------
def align_two_images_ECC(im1, im2,mode = cv2.MOTION_AFFINE):

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

