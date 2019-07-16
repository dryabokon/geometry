import cv2
import numpy
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO
import tools_image
import tools_alg_match
import tools_draw_numpy

# ---------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_manual():
    folder_input = 'images/ex_homography_manual/'
    im_scene = cv2.imread(folder_input + 'background.jpg')
    im_rect = cv2.imread(folder_input + 'rect.jpg')
    folder_output = 'images/output/'
    filename_output_rect = folder_output + 'rect_out.jpg'
    filename_output_blend = folder_output + 'background_out.jpg'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    pts_scene = numpy.array([[262, 22],[644, 135], [404, 614], [130, 531]])
    pts_rect = numpy.array([[0, 0], [300, 0], [300, 400], [0, 400]])

    #homography, status = cv2.findHomography(pts_rect, pts_scene)
    homography_afine, status = cv2.estimateAffine2D(pts_rect, pts_scene)
    homography = numpy.vstack((homography_afine,numpy.array([0,0,1])))


    #K = numpy.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]])
    #n, R, T, normal = cv2.decomposeHomographyMat(homography, K)
    #R = numpy.array(R[0])
    #T = numpy.array(T[0])
    #normal = numpy.array(normal[0])
    #HH=tools_calibrate.compose_homography(R,T,normal,K)
    #print(homography-HH)

    im_rect_out = cv2.warpPerspective(im_rect, homography, (im_scene.shape[1], im_scene.shape[0]))
    im_scene = tools_image.put_layer_on_image(im_scene, im_rect_out, background_color=(0,0,0))

    cv2.imwrite(filename_output_rect, im_rect_out)
    cv2.imwrite(filename_output_blend, im_scene)

    return


# --------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_by_keypoints(detector='SIFT', matchtype='knn'):

    folder_input = 'images/ex_keypoints/'
    img1 = cv2.imread(folder_input + 'left.jpg')
    img2 = cv2.imread(folder_input + 'rght.jpg')

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)

    folder_output = 'images/output/'
    output_filename1 = folder_output + 'left_transformed_homography.png'
    output_filename2 = folder_output + 'rght_transformed_homography.png'
    output_filename = folder_output + 'blended_homography.png'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)

    homography = tools_calibrate.get_homography_by_keypoints_desc(points1, des1, points2, des2, matchtype)
    match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)

    for each in match1:
        img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(each[1]), int(each[0]), 3, [0, 0, 255])
    for each in match2:
        img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(each[1]), int(each[0]), 3, [255, 255, 0])

    result_image1, result_image2 = tools_calibrate.get_stitched_images_using_homography(img1_gray_rgb, img2_gray_rgb,homography,background_color=(255, 255, 255))
    result_image = tools_image.blend_avg(result_image1, result_image2,background_color=(255, 255, 255))
    # result_image = tools_calibrate.blend_multi_band(result_image1, result_image2)

    cv2.imwrite(output_filename1, result_image1)
    cv2.imwrite(output_filename2, result_image2)
    cv2.imwrite(output_filename, result_image)
    return


# --------------------------------------------------------------------------------------------------------------------------
def example_03_find_translateion_by_keypoints(detector='SIFT', matchtype='knn'):

    folder_input = 'images/ex_keypoints/'
    img1 = cv2.imread(folder_input + 'left.jpg')
    img2 = cv2.imread(folder_input + 'rght.jpg')

    img1_gray_rgb = tools_image.desaturate(img1)
    img2_gray_rgb = tools_image.desaturate(img2)

    folder_output = 'images/output/'
    output_filename1 = folder_output + 'left_transformed_affine.png'
    output_filename2 = folder_output + 'rght_transformed_affine.png'
    output_filename = folder_output + 'blended_affine.png'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    #else:
        #tools_IO.remove_files(folder_output)

    points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
    points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)

    M = tools_calibrate.get_transform_by_keypoints_desc(points1, des1, points2, des2, matchtype) #(2,3)
    match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)

    for each in match1:
        img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(each[1]), int(each[0]), 3, [0, 0, 255])
    for each in match2:
        img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(each[1]), int(each[0]), 3, [255, 255, 0])

    result_image1, result_image2 = tools_calibrate.get_stitched_images_using_translation(img1_gray_rgb, img2_gray_rgb, M, background_color=(255, 255, 255))
    cv2.imwrite(output_filename1, result_image1)
    cv2.imwrite(output_filename2, result_image2)

    result_image = tools_image.blend_avg(result_image1, result_image2,background_color=(255, 255, 255))
    #result_image = tools_calibrate.blend_multi_band(result_image1, result_image2)

    cv2.imwrite(output_filename, result_image)
    return


# --------------------------------------------------------------------------------------------------------------------------


def example_03_find_translation_with_ECC(warp_mode=cv2.MOTION_TRANSLATION):
    # warp_mode = cv2.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_EUCLIDEAN
    # warp_mode = cv2.MOTION_AFFINE
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    folder_input = 'images/ex_homography_affine/'
    im1 = cv2.imread(folder_input + 'first.jpg')
    im2 = cv2.imread(folder_input + 'scond.jpg')

    folder_output = 'images/output/'
    output_filename_L1 = folder_output + 'transf1_first.jpg'
    output_filename_R1 = folder_output + 'transf1_scond.jpg'
    output_filename_L2 = folder_output + 'transf2_first.jpg'
    output_filename_R2 = folder_output + 'transf2_scond.jpg'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    if warp_mode != cv2.MOTION_HOMOGRAPHY:
        (cc, transform1) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(2, 3, dtype=numpy.float32), warp_mode, criteria)
        (cc, transform2) = cv2.findTransformECC(im2_gray, im1_gray, numpy.eye(2, 3, dtype=numpy.float32), warp_mode, criteria)
        result_image11, result_image12 = tools_calibrate.get_stitched_images_using_translation(im2, im1, transform1)
        result_image22, result_image21 = tools_calibrate.get_stitched_images_using_translation(im1, im2, transform2)
    else:
        (cc, transform1) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(3, 3, dtype=numpy.float32), warp_mode, criteria)
        (cc, transform2) = cv2.findTransformECC(im2_gray, im1_gray, numpy.eye(3, 3, dtype=numpy.float32), warp_mode, criteria)
        result_image11, result_image12 = tools_calibrate.get_stitched_images_using_homography(im2, im1, transform1)
        result_image22, result_image21 = tools_calibrate.get_stitched_images_using_homography(im1, im2, transform2)

    cv2.imwrite(output_filename_L1, result_image11)
    cv2.imwrite(output_filename_R1, result_image12)
    cv2.imwrite(output_filename_L2, result_image21)
    cv2.imwrite(output_filename_R2, result_image22)

    return


# --------------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_live():

    filename_in = 'images/ex03_live/jack.jpg'
    filename_out = 'images/output/frame.jpg'
    if os.path.exists(filename_in):
        image_model = cv2.imread(filename_in, 0)
    else:
        image_model=numpy.zeros((480,640),dtype=numpy.uint8)
    h, w = image_model.shape

    capture = cv2.VideoCapture(0)

    while (True):
        ret, image_captr = capture.read()

        key = cv2.waitKey(1)

        if key & 0xFF == 27:
            break
        if (key & 0xFF == 13) or (key & 0xFF == 32):
            cv2.imwrite(filename_out,image_captr)

        points1, des1 = tools_alg_match.get_keypoints_desc(image_model, 'ORB',)
        points2, des2 = tools_alg_match.get_keypoints_desc(image_captr, 'ORB',)
        homography = tools_calibrate.get_homography_by_keypoints_desc(points1, des1, points2, des2, 'knn')


        if (homography== []):
            pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography)
            img2 = cv2.polylines(image_captr, [numpy.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', img2)
            print(homography)
        else:
            cv2.imshow('frame', image_captr)

    capture.release()
    cv2.destroyAllWindows()
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    xxx()
    #example_03_find_homography_manual()
    #example_03_find_homography_by_keypoints('ORB')
    #example_03_find_translateion_by_keypoints('ORB')

    #example_03_find_translation_with_ECC()
    #example_03_find_homography_live()


