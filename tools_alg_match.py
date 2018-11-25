# ---------------------------------------------------------------------------------------------------------------------
import os
import numpy
import cv2
import math
from scipy.misc import toimage
from scipy import ndimage
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image


# ---------------------------------------------------------------------------------------------------------------------
def calc_hit_field(image_base, image_pattern):
    x1 = int(1 * image_pattern.shape[0] / 4)
    x2 = int(3 * image_pattern.shape[0] / 4)
    y1 = int(1 * image_pattern.shape[1] / 4)
    y2 = int(3 * image_pattern.shape[1] / 4)

    hitmap1 = calc_hit_field_basic(image_base, image_pattern).astype(numpy.float)
    hitmap2 = calc_hit_field_basic(image_base, image_pattern[x1:x2, y1:y2]).astype(numpy.float)
    hitmap3 = calc_hit_field_basic(image_base, image_pattern[x1:x2, :]).astype(numpy.float)
    hitmap4 = calc_hit_field_basic(image_base, image_pattern[:, y1:y2]).astype(numpy.float)

    hitmap = (hitmap4.copy()).astype(numpy.float)
    hitmap = hitmap1 * hitmap2 * hitmap3 * hitmap4

    N = int(image_pattern.shape[0] * 0.25)
    hitmap = ndimage.convolve(hitmap, numpy.ones((N, N)), mode='constant')
    max = numpy.max(hitmap) / 255
    hitmap /= max

    return hitmap.astype(numpy.uint8)


# ---------------------------------------------------------------------------------------------------------------------
def calc_hit_field_basic(image_base, image_pattern):
    hitmap_gray = cv2.matchTemplate(image_base, image_pattern, method=cv2.TM_CCOEFF_NORMED)

    min = hitmap_gray.min()
    max = hitmap_gray.max()
    hitmap_gray = (225 * (hitmap_gray - min) / (max - min)).astype(numpy.uint8)
    hitmap_gray = numpy.reshape(hitmap_gray, (hitmap_gray.shape[0], hitmap_gray.shape[1], 1))

    hitmap_gray2 = numpy.zeros((image_base.shape[0], image_base.shape[1], 1)).astype(numpy.uint8)
    shift_x = int((image_base.shape[0] - hitmap_gray.shape[0]) / 2)
    shift_y = int((image_base.shape[1] - hitmap_gray.shape[1]) / 2)
    hitmap_gray2[shift_x:shift_x + hitmap_gray.shape[0], shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[:, :, 0]

    hitmap_gray2[:shift_x, shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[0, :, 0]
    hitmap_gray2[shift_x + hitmap_gray.shape[0]:, shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[-1, :, 0]

    for row in range(0, hitmap_gray2.shape[0]):
        hitmap_gray2[row, :shift_y, 0] = hitmap_gray2[row, shift_y, 0]
        hitmap_gray2[row, shift_y + hitmap_gray.shape[1]:, 0] = hitmap_gray2[row, shift_y + hitmap_gray.shape[1] - 1, 0]

    hitmap_2d = numpy.reshape(hitmap_gray2, (hitmap_gray2.shape[0], hitmap_gray2.shape[1]))

    return hitmap_2d


# ---------------------------------------------------------------------------------------------------------------------
def rolling_std(gray, drow, dcol):
    g1_summ = numpy.cumsum(numpy.cumsum(gray, axis=0), axis=1)
    g2_summ = numpy.cumsum(numpy.cumsum(numpy.multiply(gray, gray), axis=0), axis=1)

    weights = numpy.zeros(((2 + 2 * drow), (2 + 2 * dcol)))
    weights[0, 0] = +1
    weights[-1, -1] = +1
    weights[0, -1] = -1
    weights[-1, 0] = -1

    summ = ndimage.convolve(g1_summ, weights, mode='constant', origin=((-1, -1)))
    summ2 = ndimage.convolve(g2_summ, weights, mode='constant', origin=((-1, -1)))
    summ[:, gray.shape[1] - dcol:] = 0
    summ[gray.shape[0] - drow:, :] = 0
    summ2[:, gray.shape[1] - dcol:] = 0
    summ2[gray.shape[0] - drow:, :] = 0

    norm = ((1 + 2 * drow) * (1 + 2 * dcol))
    res = summ2 / norm - (summ / norm) * (summ / norm)
    res = numpy.sqrt(res)

    res = tools_image.fill_border(res, drow, dcol, res.shape[0] - drow, res.shape[1] - dcol)

    return res


# ---------------------------------------------------------------------------------------------------------------------
def save_quality_to_disk(result_4d, disp_v1, disp_v2, disp_h1, disp_h2, path_debug):
    if path_debug != '' and (not os.path.exists(path_debug)):
        os.makedirs(path_debug)
    else:
        tools_IO.remove_files(path_debug)

    result_3d = numpy.min(result_4d, axis=2)

    tools_IO.remove_files(path_debug)

    for dh in range(disp_h1, disp_h2):
        hitmap_2d = result_3d[:, :, dh - disp_h1].astype(numpy.uint8)
        hitmap_RGB_gre, hitmap_RGB_jet = tools_image.hitmap2d_to_jet(hitmap_2d)
        toimage(hitmap_RGB_jet).save(path_debug + ('q1_%03d.bmp' % dh))

    result_3d = numpy.min(result_4d, axis=2)
    for row in range(0, result_4d.shape[0]):
        hitmap_2d = result_3d[row, :, :].astype(numpy.uint8)
        hitmap_RGB_gre, hitmap_RGB_jet = tools_image.hitmap2d_to_jet(hitmap_2d.T)
        toimage(hitmap_RGB_jet).save(path_debug + ('q2_%03d.bmp' % row))

    return


# ---------------------------------------------------------------------------------------------------------------------
def calc_match_quality(image1, image2, disp_v1, disp_v2, disp_h1, disp_h2, window_size=5, path_debug=''):
    if len(image1.shape) != 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if len(image2.shape) != 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    result_4d = numpy.full((image1.shape[0], image1.shape[1], disp_v2 - disp_v1, disp_h2 - disp_h1), -1)

    for dh in range(disp_h1, disp_h2):
        if dh == -15:
            dh = dh
        for dv in range(disp_v1, disp_v2):
            gray2_shifted = tools_image.shift_image(image2, -dv, -dh)
            gray = 127 + (image1 - gray2_shifted) / 2

            hitmap_2d = rolling_std(gray, window_size, window_size)
            result_4d[:, :, dv - disp_v1, dh - disp_h1] = hitmap_2d

    min = numpy.min(result_4d)
    result_4d -= min
    max = numpy.max(result_4d)
    result_4d = 255 - (255 * result_4d / max).astype(numpy.uint8)

    if path_debug != '':
        save_quality_to_disk(result_4d, disp_v1, disp_v2, disp_h1, disp_h2, path_debug)

    return result_4d


# ---------------------------------------------------------------------------------------------------------------------
def visualize_matches_map(disp_col, disp_v1, disp_v2, disp_h1, disp_h2):
    image_disp_hor = numpy.zeros((disp_col.shape[0], disp_col.shape[1])).astype(numpy.float32)
    image_disp_hor[:, :] = disp_col[:, :]

    image_disp_hor = (image_disp_hor - disp_h1) * 255 / (disp_h2 - disp_h1)
    image_disp_hor_RGB_gre, image_disp_hor_RGB_jet = tools_image.hitmap2d_to_jet(image_disp_hor.astype(numpy.uint8))
    # image_disp_hor_RGB_gre, image_disp_hor_RGB_jet = tools_image.hitmap2d_to_viridis(image_disp_hor.astype(numpy.uint8))

    image_disp_hor_RGB_jet[disp_col >= disp_h2 + 1] = [128, 128, 128]
    image_disp_hor_RGB_jet[disp_col <= disp_h1 - 1] = [128, 128, 128]

    return image_disp_hor_RGB_jet


# ---------------------------------------------------------------------------------------------------------------------
def calc_matches_mat(quality_4d, disp_v1, disp_v2, disp_h1, disp_h2):
    quality_3d = numpy.max(quality_4d, axis=3)
    disp_row = numpy.argmax(quality_3d, axis=2) + disp_v1

    quality_3d = numpy.max(quality_4d, axis=2)
    disp_col = numpy.argmax(quality_3d, axis=2) + disp_h1

    return disp_row, disp_col


# ---------------------------------------------------------------------------------------------------------------------
def get_keypoints_desc(image1, detector='SIFT'):
    if len(image1.shape) == 2:
        im1_gray = image1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if detector == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    elif detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:  # detector == 'ORB'
        detector = cv2.ORB_create()

    kp, desc = detector.detectAndCompute(im1_gray, None)

    points = []
    if (len(kp) > 0):
        points = numpy.array([kp[int(idx)].pt for idx in range(0, len(kp))]).astype(int)

    return points, desc


# ---------------------------------------------------------------------------------------------------------------------
def get_keypoints_STAR(image1):
    if len(image1.shape) == 2:
        im1_gray = image1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    detector = cv2.xfeatures2d.StarDetector_create()

    kp = detector.detect(image1)

    points = []
    if (len(kp) > 0):
        points = numpy.array([kp[int(idx)].pt for idx in range(0, len(kp))]).astype(int)

    return points

# ---------------------------------------------------------------------------------------------------------------------
def get_corners_Fast(image1):
    if len(image1.shape) == 2:
        im1_gray = image1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    detector = cv2.FastFeatureDetector_create()
    kp = detector.detect(image1)

    points = []
    if (len(kp) > 0):
        points = numpy.array([kp[int(idx)].pt for idx in range(0, len(kp))]).astype(int)

    return points


# ---------------------------------------------------------------------------------------------------------------------
def get_corners_Shi_Tomasi(image1):
    if len(image1.shape) == 2:
        im1_gray = image1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    detector = cv2.FastFeatureDetector_create()

    corners = cv2.goodFeaturesToTrack(im1_gray, 25, 0.01, 10)
    corners = corners.reshape(corners.shape[0], corners.shape[2]).astype(int)

    return corners


# ---------------------------------------------------------------------------------------------------------------------
def get_corners_Harris(image1):
    if len(image1.shape) == 2:
        im1_gray = image1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = numpy.float32(im1_gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # cv2.imwrite(filename_out2, 255*(dst-numpy.min(dst))/(numpy.max(dst)-numpy.min(dst)))

    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    # cv2.imwrite(filename_out3, 255 * (dst - numpy.min(dst)) / (numpy.max(dst) - numpy.min(dst)))
    dst = numpy.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, numpy.float32(centroids), (5, 5), (-1, -1), criteria).astype(int)
    centroids = centroids.astype(int)
    return corners


# ---------------------------------------------------------------------------------------------------------------------
def get_disparity_v_01(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2):
    window_size = 7
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=disp_h1,
        numDisparities=int(0 + (disp_h2 - disp_h1) / 16) * 16,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=disp_h2,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    dispr = right_matcher.compute(imgR, imgL)
    displ = left_matcher.compute(imgL, imgR)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    filteredImg_L = wls_filter.filter(displ, imgL, None, dispr)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=right_matcher)
    filteredImg_R = wls_filter.filter(dispr, imgR, None, displ)

    return filteredImg_L / 16, -filteredImg_R / 16


# ---------------------------------------------------------------------------------------------------------------------
def get_disparity_v_02(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2):
    max = numpy.maximum(math.fabs(disp_h1), math.fabs(disp_h2))
    levels = int(1 + (max) / 16) * 16
    stereo = cv2.StereoBM_create(numDisparities=levels, blockSize=15)
    displ = stereo.compute(imgL, imgR)

    dispr = numpy.flip(stereo.compute(numpy.flip(imgR, axis=1), numpy.flip(imgL, axis=1)), axis=1)
    return -displ / 16, -dispr / 16
# ---------------------------------------------------------------------------------------------------------------------
