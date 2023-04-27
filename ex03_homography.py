import cv2
import numpy
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
from CV import tools_calibrate
from CV import tools_alg_match
from CV import tools_pr_geom
# ---------------------------------------------------------------------------------------------------------------------
def example_02_homography_manual():

    im_source = cv2.imread('./images/ex_homography_manual/03/source.jpg')
    im_target = cv2.imread('./images/ex_homography_manual/03/target.jpg')
    pad = 0
    background_color = (0, 0, 0)

    # p_source = numpy.array([[0+pad, 0+pad],[im_source.shape[1]-pad, pad],[im_source.shape[1]-pad, im_source.shape[0]-pad],[pad, im_source.shape[0]-pad]],dtype=numpy.float32)
    # p_target = numpy.array([[1012, 276], [1830, 345], [1800, 989], [999, 865]])

    p_source = numpy.array([[700,264],[708,912],[418,880],[410,590]],dtype=numpy.float32)
    p_target = numpy.array([[443,88], [442, 487], [171, 478], [173, 236]],dtype=numpy.float32)


    homography2, status = tools_pr_geom.fit_euclid(p_source.reshape((-1,1,2)), p_target.reshape((-1,1,2)))
    image_trans = cv2.warpAffine(im_source, homography2, (im_target.shape[1], im_target.shape[0]),borderValue=background_color)

    # homography2, status = tools_pr_geom.fit_homography(p_source.reshape((-1, 1, 2)), p_target.reshape((-1, 1, 2)))
    # image_trans = cv2.warpPerspective(im_source, homography2, (im_target.shape[1], im_target.shape[0]),borderValue=background_color)

    #cv2.imwrite(filename_out, image_trans)
    result = tools_image.put_layer_on_image(im_target,image_trans,background_color = background_color)
    cv2.imwrite('./images/output/homohraphy_result2.png', result)
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_03_find_homography_manual():

    folder_output = 'images/output/'

    # im_source = cv2.imread('./images/ex_homography_manual_gis/05/kad1_small.jpg')
    # im_target = cv2.imread('./images/ex_homography_manual_gis/05/map3.jpg')
    # p_source = numpy.array([[716, 633], [1468, 148], [1698, 896], [1242, 1210]])
    # p_target = numpy.array([[212, 366], [671, 80], [801, 526], [511, 715]])

    im_source = cv2.imread('./images/ex_homography_manual/03/source.jpg')
    im_target = cv2.imread('./images/ex_homography_manual/03/target.jpg')
    p_source = numpy.array([[192, 246], [897, 240], [909, 1194], [189, 1179]], dtype=numpy.float32)
    p_target = numpy.array([[217, 212], [409, 194], [408, 420], [215, 417]], dtype=numpy.float32)

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)
    im_source_gray = tools_draw_numpy.draw_points(tools_image.desaturate(im_source), p_source, color=(0, 0, 255), w=24, put_text=False)
    #im_source_gray = im_source.copy()
    cv2.imwrite(folder_output+'im_source.png',im_source_gray)
    cv2.imwrite(folder_output+'im_target.png',tools_draw_numpy.draw_points(tools_image.desaturate(im_target), p_target, color=(0, 0, 255), w=4, put_text=False))

    im_source = im_source_gray

    # via affine
    homography_afine, status = tools_pr_geom.fit_affine(p_source, p_target)
    homography = numpy.vstack((homography_afine,numpy.array([0,0,1])))
    image_affine = tools_image.put_layer_on_image(im_target,cv2.warpPerspective(im_source, homography, (im_target.shape[1], im_target.shape[0])),background_color=(0, 0, 0))
    cv2.imwrite(folder_output+'affine.png', image_affine)

    # homography itself
    homography2, status = tools_pr_geom.fit_homography(p_source.reshape((-1,1,2)), p_target.reshape((-1,1,2)))
    image_homo = tools_image.put_layer_on_image(im_target,cv2.warpPerspective(im_source, homography2, (im_target.shape[1], im_target.shape[0])),background_color=(0, 0, 0))
    cv2.imwrite(folder_output + 'homography.png', image_homo)

    # homography normalized
    K = numpy.array([[im_target.shape[1], 0, 0], [0, im_target.shape[0], 0], [0, 0, 1]])
    n, R, T, normal = cv2.decomposeHomographyMat(homography, K)
    R = numpy.array(R[0])
    T = numpy.array(T[0])
    normal = numpy.array(normal[0])
    homography3 = tools_calibrate.compose_homography(R, T, normal, K)
    image_homo2 = tools_image.put_layer_on_image(im_target, cv2.warpPerspective(im_source, homography3,(im_target.shape[1], im_target.shape[0])),background_color=(0, 0, 0))
    cv2.imwrite(folder_output + 'homography_normalized.png',image_homo2 )



    return
# --------------------------------------------------------------------------------------------------------------------------
def example_04_find_homography_by_keypoints(detector='SIFT', matchtype='knn'):

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
def example_05_find_translation_by_keypoints(detector='SIFT', matchtype='knn'):

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
def example_06_find_translation_with_ECC(warp_mode=cv2.MOTION_TRANSLATION):
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
def example_07_find_homography_live():
    USE_CAMERA = False
    USE_TRANSFORM = True

    filename_out = './images/output/frame.jpg'
    image_deck = cv2.imread('./images/ex_homography_live/frame2.jpg')
    image_card = cv2.imread('./images/ex_homography_live/card.jpg')
    image_sbst = cv2.imread('./images/ex_homography_live/card2.jpg')

    image_sbst = cv2.resize(image_sbst,(image_card.shape[1],image_card.shape[0]))

    points1, des1 = tools_alg_match.get_keypoints_desc(image_card, 'SURF')
    if USE_CAMERA:
        capture = cv2.VideoCapture(0)

    while (True):
        if USE_CAMERA:
            ret, image_deck = capture.read()

        points2, des2 = tools_alg_match.get_keypoints_desc(image_deck, 'SURF')

        if USE_TRANSFORM:
            H = tools_calibrate.get_transform_by_keypoints_desc(points1, des1, points2, des2, 'knn')
        else:
            H = tools_calibrate.get_homography_by_keypoints_desc(points1, des1, points2, des2, 'knn')


        if (H is not None):
            if USE_TRANSFORM:
                aligned1, aligned2 = tools_calibrate.get_stitched_images_using_translation(image_sbst, image_deck, H, background_color=(0, 0, 0),keep_shape=True)
            else:
                aligned1, aligned2 = tools_calibrate.get_stitched_images_using_homography(image_sbst, image_deck, H,background_color=(0, 0, 0))
            #im_result = tools_image.put_layer_on_image(aligned2,aligned1,background_color=(0,0,0))
            im_result = tools_image.blend_multi_band_large_small(aligned2,aligned1, background_color=(0, 0, 0), filter_size=10,leveln_default=1)
            cv2.imshow('frame', im_result)
        else:
            cv2.imshow('frame', image_deck)


        key = cv2.waitKey(1)
        if key & 0xFF == 27:break
        if (key & 0xFF == 13) or (key & 0xFF == 32):cv2.imwrite(filename_out,image_deck)

    if USE_CAMERA:
        capture.release()

    cv2.destroyAllWindows()
    return

# ---------------------------------------------------------------------------------------------------------------------
def example_08_auto_translate(folder_in,folder_out):

    filenames = tools_IO.get_filenames(folder_in,'*.png')
    img_base =  cv2.imread(folder_in+filenames[0])
    cv2.imwrite(folder_out + filenames[0], img_base)
    points_base, des_base = tools_alg_match.get_keypoints_desc(img_base)

    for filename in filenames[1:]:
        img_cand = cv2.imread(folder_in + filename)
        points_cand, des_cand = tools_alg_match.get_keypoints_desc(img_cand)
        M = tools_calibrate.get_transform_by_keypoints_desc(points_cand, des_cand, points_base, des_base)
        result_image2, result_image1 = tools_calibrate.get_stitched_images_using_translation(img_cand, img_base, M)
        cv2.imwrite(folder_out+filename,result_image2)

    return
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_03_find_homography_manual()
