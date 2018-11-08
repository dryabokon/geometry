import cv2
import numpy
import os
import tools_calibrate
import tools_IO
import math
#--------------------------------------------------------------------------------------------------------------------------
def example_01_calibrate_camera():

    filename_input_grid     = '_images/ex01/grid.jpg'
    folder_input_chess      = '_images/ex01/left/'
    filename_input_chess    = folder_input_chess + 'left01.jpg'

    folder_output           = '_images/ex01_out/'
    filename_output_chess   = folder_output + 'undistorted_chess.jpg'
    filename_output_grid    = folder_output + 'undistorted_grid.jpg'
    chess_rows=6
    chess_cols=7

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    cameraMatrix, dist = tools_calibrate.get_proj_dist_mat_for_images(folder_input_chess,chess_rows,chess_cols)
    undistorted_chess  = tools_calibrate.undistort_image(cameraMatrix, dist, filename_input_chess)
    undistorted_grid   = tools_calibrate.undistort_image(cameraMatrix, dist, filename_input_grid)
    cv2.imwrite(filename_output_chess, undistorted_chess)
    cv2.imwrite(filename_output_grid , undistorted_grid)

    return
# -------------------------------------------------------------------------------------------------------------------------
def example_02_rectify_pair():

    folder_input_chess = '_images/ex01/left/'
    folder_input       = '_images/ex01/'
    filename_input_1   = folder_input + 'left/left01.jpg'
    filename_input_2   = folder_input + 'right/right01.jpg'
    folder_output      = '_images/ex01_out/'
    filename_output_1  = folder_output + 'left01_rect.jpg'
    filename_output_2  = folder_output + 'right01_rect.jpg'

    chess_rows=6
    chess_cols=7

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

        cameraMatrix, dist = tools_calibrate.get_proj_dist_mat_for_images(folder_input_chess, chess_rows, chess_cols)

    im1_remapped, im2_remapped = tools_calibrate.rectify_pair(cameraMatrix, dist, filename_input_1, filename_input_2, chess_rows, chess_cols)
    cv2.imwrite(filename_output_1, im1_remapped)
    cv2.imwrite(filename_output_2, im2_remapped)

    return

#==========================================================================================================================
def example_03_find_homography_manual():

    im_scene                = cv2.imread('D:/Projects/Cyb/Stereo/ex03_manual/scene.jpg')
    im_rect                 = cv2.imread('D:/Projects/Cyb/Stereo/ex03_manual/book.jpg')
    filename_output_rect    =            'D:/Projects/Cyb/Stereo/ex03_manual/book_out.jpg'
    filename_output_blend   =            'D:/Projects/Cyb/Stereo/ex03_manual/scene_out.jpg'

    pts_scene = numpy.array([[237, 193], [400, 279], [236, 501], [55, 355]])
    pts_rect  = numpy.array([[0  , 0]  , [300, 0]  , [300, 400], [0 , 400]])

    homography, status = cv2.findHomography(pts_rect,pts_scene)
    im_rect_out = cv2.warpPerspective(im_rect, homography, (im_scene.shape[1], im_scene.shape[0]))
    im_scene = Calibrate.put_layer_on_image(im_scene, im_rect_out)

    cv2.imwrite(filename_output_rect, im_rect_out)
    cv2.imwrite(filename_output_blend, im_scene)

    return
#--------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_BFMatcher():
    img_source 		    = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography/left.jpg')
    img_destin 		    = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography/rght.jpg')
    output_filename_AL    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneA_left.png'
    output_filename_AR    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneA_rght.png'
    output_filename_BL    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneB_left.png'
    output_filename_BR    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneB_rght.png'

    homography,result_image1 = Calibrate.get_BF_homography(img_destin, img_source)
    #cv2.imwrite(output_matches, result_image1)
    result_image1,result_image2 = Calibrate.get_stitched_images(img_destin, img_source, homography)
    cv2.imwrite(output_filename_AL, result_image1)
    cv2.imwrite(output_filename_AR, result_image2)

    homography,result_image1 = Calibrate.get_BF_homography(img_source, img_destin)
    result_image1,result_image2 = Calibrate.get_stitched_images(img_source, img_destin, homography)
    cv2.imwrite(output_filename_AL, result_image1)
    cv2.imwrite(output_filename_AR, result_image2)
    return
#--------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_SIFT():
    img_source 		    = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography/left.jpg')
    img_destin 		    = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography/rght.jpg')
    output_filename_AL    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneA_left.png'
    output_filename_AR    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneA_rght.png'
    output_filename_BL    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneB_left.png'
    output_filename_BR    =          'D:/Projects/Cyb/Stereo/ex03_homography/sceneB_rght.png'


    homography=Calibrate.get_sift_homography(img_source, img_destin)
    result_image1,result_image2 = Calibrate.get_stitched_images(img_destin, img_source, homography,background=(0, 0, 0))
    cv2.imwrite(output_filename_AL, result_image1)
    cv2.imwrite(output_filename_AR, result_image2)

    homography = Calibrate.get_sift_homography(img_destin, img_source)
    result_image1,result_image2 = Calibrate.get_stitched_images(img_source, img_destin, homography,background=(0, 0, 0))
    cv2.imwrite(output_filename_BR, result_image1)
    cv2.imwrite(output_filename_BL, result_image2)
    return
#--------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_ECC():

    #im1 = cv2.imread(       'D:/Projects/Cyb/Stereo/ex03_homo_affine/image01.bmp')
    #im2 = cv2.imread(       'D:/Projects/Cyb/Stereo/ex03_homo_affine/image02.bmp')

    im1 = cv2.imread(       'D:/Projects/Cyb/Stereo/ex03_homo_affine/000.bmp')
    im2 = cv2.imread(       'D:/Projects/Cyb/Stereo/ex03_homo_affine/007.bmp')
    output_filename_A1_T =  'D:/Projects/Cyb/Stereo/ex03_homo_affine/image01_aligned_transf.bmp'
    output_filename_A2_T =  'D:/Projects/Cyb/Stereo/ex03_homo_affine/image02_aligned_transf.bmp'
    output_filename_A1_H =  'D:/Projects/Cyb/Stereo/ex03_homo_affine/image01_aligned_homogr.bmp'
    output_filename_A2_H =  'D:/Projects/Cyb/Stereo/ex03_homo_affine/image02_aligned_homogr.bmp'

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    sz = im1.shape

    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    warp_mode = cv2.MOTION_TRANSLATION
    (cc, warp_matrix12) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(2, 3, dtype=numpy.float32), warp_mode,criteria)
    (cc, warp_matrix21) = cv2.findTransformECC(im2_gray, im1_gray, numpy.eye(2, 3, dtype=numpy.float32), warp_mode,criteria)
    im2_aligned = cv2.warpAffine(im2, warp_matrix12, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    im1_aligned = cv2.warpAffine(im1, warp_matrix21, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite(output_filename_A1_T, im1_aligned)
    cv2.imwrite(output_filename_A2_T, im2_aligned)

    '''
    warp_mode = cv2.MOTION_HOMOGRAPHY
    (cc, warp_matrix12) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(3, 3, dtype=numpy.float32), warp_mode, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10))
    (cc, warp_matrix21) = cv2.findTransformECC(im2_gray, im1_gray, numpy.eye(3, 3, dtype=numpy.float32), warp_mode, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10))
    im2_aligned = cv2.warpPerspective(im2, warp_matrix12, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    im1_aligned = cv2.warpPerspective(im1, warp_matrix21, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite(output_filename_A1_H, im1_aligned)
    cv2.imwrite(output_filename_A2_H, im2_aligned)
    '''

    return
#--------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_multiple():

    return
# --------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_live():

    MIN_MATCHES = 15
    image_model = cv2.imread('D:/Projects/Cyb/Stereo/ex03_live/card.jpg', 0)
    h, w = image_model.shape

    capture = cv2.VideoCapture(0)

    while (True):
        ret, image_captr = capture.read()
        if cv2.waitKey(1) & 0xFF == 27:
            break

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_model, des_model = orb.detectAndCompute(image_model, None)
        kp_frame, des_frame = orb.detectAndCompute(image_captr, None)
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = numpy.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)


        pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(image_captr, [numpy.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', img2)

    cap.release()
    cv2.destroyAllWindows()

#==========================================================================================================================

def example_04_blend_avg():
    img1 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneA_left.png')
    img2 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneA_rght.png')
    cv2.imwrite(      'D:/Projects/Cyb/Stereo/ex04_blend/sceneA_mix_avg.png', Calibrate.blend_avg(img1, img2,(0,0,0)))

    img1 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneB_left.png')
    img2 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneB_rght.png')
    cv2.imwrite(      'D:/Projects/Cyb/Stereo/ex04_blend/sceneB_mix_avg.png', Calibrate.blend_avg(img1, img2,(255,255,255)))
    return
#--------------------------------------------------------------------------------------------------------------------------
def example_04_blend_multi_band():
    img1 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneA_left.png')
    img2 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneA_rght.png')
    cv2.imwrite(      'D:/Projects/Cyb/Stereo/ex04_blend/sceneA_mix_multi.png', Calibrate.blend_multi_band(img1, img2,(0,0,0)))

    img1 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneB_left.png')
    img2 = cv2.imread('D:/Projects/Cyb/Stereo/ex04_blend/sceneB_rght.png')
    cv2.imwrite(      'D:/Projects/Cyb/Stereo/ex04_blend/sceneB_mix_multi.png', Calibrate.blend_multi_band(img1, img2,(255,255,255)))
    return
#--------------------------------------------------------------------------------------------------------------------------
def example_04_find_homography_blend_multi_band():

    img_left  = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography_blend_multi_band/left.jpg')
    img_right = cv2.imread('D:/Projects/Cyb/Stereo/ex03_homography_blend_multi_band/rght.jpg')
    output_left          = 'D:/Projects/Cyb/Stereo/ex03_homography_blend_multi_band/left_res.jpg'
    output_right         = 'D:/Projects/Cyb/Stereo/ex03_homography_blend_multi_band/rght_res.jpg'

    homography = Calibrate.get_sift_homography(img_right, img_left)
    result_right, result_left = Calibrate.get_stitched_images(img_left, img_right, homography)
    result_image = Calibrate.blend_multi_band(result_left, result_right)
    cv2.imwrite(output_left, result_image)

    homography = Calibrate.get_sift_homography(img_left, img_right)
    result_left, result_right  = Calibrate.get_stitched_images(img_right, img_left, homography)
    result_image = Calibrate.blend_multi_band(result_left, result_right)
    cv2.imwrite(output_right, result_image)
    return

#==========================================================================================================================
def example_05_derive_transform():
    img1 = cv2.imread('D:/Projects/Cyb/Stereo/ex05_derive_transform/base.png')
    img2 = cv2.imread('D:/Projects/Cyb/Stereo/ex05_derive_transform/deform.png')
    output_filename1 = 'D:/Projects/Cyb/Stereo/ex05_derive_transform/res1.png'
    output_filename2 = 'D:/Projects/Cyb/Stereo/ex05_derive_transform/res2.png'
    R, T, normal, HH, K = Calibrate.derive_transform(img1,img2)
    return R, T, normal, HH, K
# --------------------------------------------------------------------------------------------------------------------------
def example_05_animage_homography():
    img_left  = cv2.imread('D:/Projects/Cyb/Stereo/ex05_homo_middle/left.jpg')
    img_rght  = cv2.imread('D:/Projects/Cyb/Stereo/ex05_homo_middle/rght.jpg')
    output_filename     = 'D:/Projects/Cyb/Stereo/ex05_homo_middle/res'
    R, T, normal, H, K = Calibrate.derive_transform(img_left, img_rght)

    for alpha in range(0,20):
        HH = Calibrate.bend_homography(H, alpha/20.0)
        result_right, result_left = Calibrate.get_stitched_images(img_rght, img_left, HH)
        cv2.imwrite((output_filename+('%02d.png' % alpha)),result_right )

    return
# --------------------------------------------------------------------------------------------------------------------------
def example_05_find_homography_bend():

    img_left  = cv2.imread('D:/Projects/Cyb/Stereo/ex05_homo_middle/left.jpg')
    img_rght  = cv2.imread('D:/Projects/Cyb/Stereo/ex05_homo_middle/rght.jpg')
    output    =            'D:/Projects/Cyb/Stereo/ex05_homo_middle/res.jpg'
    output1   =            'D:/Projects/Cyb/Stereo/ex05_homo_middle/res1.jpg'
    output2   =            'D:/Projects/Cyb/Stereo/ex05_homo_middle/res2.jpg'


    #R, T, normal, H, K = Calibrate.derive_transform(img_left, img_rght)
    #left_coord  = numpy.float32([[377,237],[469,237],[471,470],[377,374]]).reshape(-1, 1, 2)
    #right_coord = numpy.float32([[86, 177],[179,178],[179,269],[80, 267]]).reshape(-1, 1, 2)

    #B  = Calibrate.bend_homography(H, 0.5)
    #B2 = Calibrate.get_inverse_homography(Calibrate.bend_homography(H, 0.5))
    #print(numpy.dot(B,B2))
    #print(numpy.dot(B2,B))
    #print()

    #left_coord_transf = cv2.perspectiveTransform(left_coord, B)
    #print(left_coord_transf.reshape(4, 2))
    #right_coord_transf = cv2.perspectiveTransform(right_coord, B2)
    #print(right_coord_transf.reshape(4,2))


    H_left,H_right = Calibrate.get_homography_middle(img_left, img_rght)
    result_left,result_right = Calibrate.get_stitched_images_middle(img_left,img_rght, H_left,H_right)
    result_image = Calibrate.blend_multi_band(result_left, result_right)
    cv2.imwrite(output1, result_left)
    cv2.imwrite(output2, result_right)
    cv2.imwrite(output, result_image)

    return
#==========================================================================================================================
def example_XX_pose_estimation(foldername,cameraMatrix, dist):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = numpy.zeros((6 * 7, 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:7, 0:6].T.reshape(-1, 2)
    axis = numpy.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)


    for fname in glob(foldername + 'left*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, cameraMatrix, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)

            img = Calibrate.draw_corners(img, corners2, imgpts)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    return
#==========================================================================================================================
if __name__ == '__main__':

    numpy.set_printoptions(suppress=True, precision=0)
    #example_01_calibrate_camera()
    example_02_rectify_pair()

    #example_03_find_homography_manual()
    #example_03_find_homography_BFMatcher()
    #example_03_find_homography_SIFT()
    #example_03_find_homography_ECC()
    #example_03_find_homography_live()
    #example_03_find_homography_multiple()


    #example_04_blend_avg()
    #example_04_blend_multi_band()
    #example_04_find_homography_blend_multi_band()


    #example_05_derive_transform()
    #example_05_find_homography_bend()
    #example_05_animage_homography()
