# ---------------------------------------------------------------------------------------------------------------------
import os
import numpy
import cv2
import math
from scipy.misc import toimage
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
import tools_calibrate
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
    #hitmap = ndimage.uniform_filter(hitmap, size=(N,N), mode='constant')
    min = numpy.min(hitmap)
    max = numpy.max(hitmap)
    hitmap -=min
    hitmap /= (max-min)/255
    hitmap = hitmap.astype(numpy.uint8)

    hitmap[hitmap==0]=1

    return hitmap


# ---------------------------------------------------------------------------------------------------------------------
def calc_hit_field_basic(image_base, image_pattern, rotation_tol = 0, rotation_step = 0):

    #method = cv2.TM_SQDIFF_NORMED
    #method = cv2.TM_CCORR
    #method = cv2.TM_CCORR_NORMED
    #method = cv2.TM_CCOEFF
    method = cv2.TM_CCOEFF_NORMED

    hitmap_gray = cv2.matchTemplate(image_base, image_pattern, method=method)
    min = hitmap_gray.min()
    max = hitmap_gray.max()
    hitmap_gray = (225 * (hitmap_gray - min) / (max - min)).astype(numpy.uint8)
    hitmap_2d = tools_image.canvas_extrapolate_gray(hitmap_gray,image_base.shape[0], image_base.shape[1])

    if rotation_tol != 0 and rotation_step != 0:
        for angle in numpy.arange(-rotation_tol,+rotation_tol,rotation_step):
            M = cv2.getRotationMatrix2D((image_pattern.shape[1]/2, image_pattern.shape[1]/2), angle, 1)
            image_pattern_rotated = cv2.warpAffine(image_pattern, M, (image_pattern.shape[1], image_pattern.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            hitmap_gray = cv2.matchTemplate(image_base, image_pattern_rotated, method=method)
            min = hitmap_gray.min()
            max = hitmap_gray.max()
            hitmap_gray = (225 * (hitmap_gray - min) / (max - min)).astype(numpy.uint8)
            hitmap_gray = tools_image.canvas_extrapolate_gray(hitmap_gray, image_base.shape[0], image_base.shape[1])
            hitmap_2d = numpy.maximum(hitmap_2d,hitmap_gray)

    return hitmap_2d


# ---------------------------------------------------------------------------------------------------------------------
def visualize_matches_map(disp_col, disp_v1, disp_v2, disp_h1, disp_h2):
    image_disp_hor = numpy.zeros((disp_col.shape[0], disp_col.shape[1])).astype(numpy.float32)
    image_disp_hor[:, :] = disp_col[:, :]

    image_disp_hor = (image_disp_hor - disp_h1) * 255 / (disp_h2 - disp_h1)
    image_disp_hor_RGB_jet = tools_image.hitmap2d_to_jet(-image_disp_hor.astype(numpy.uint8))
    # image_disp_hor_RGB_gre, image_disp_hor_RGB_jet = tools_image.hitmap2d_to_viridis(image_disp_hor.astype(numpy.uint8))

    image_disp_hor_RGB_jet[disp_col >= disp_h2 + 1] = [128, 128, 128]
    image_disp_hor_RGB_jet[disp_col <= disp_h1 - 1] = [128, 128, 128]

    return image_disp_hor_RGB_jet

# ---------------------------------------------------------------------------------------------------------------------
def visualize_matches_coord(coord1,coord2,gray1_rgb,gray2_rgb):
    for i in range(0, coord1.shape[0]):
        r = int(255 * numpy.random.rand())
        color = cv2.cvtColor(numpy.array([r, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)
        #gray1_rgb = tools_draw_numpy.draw_circle(gray1_rgb, coord1[i, 1], coord1[i, 0], 4, color)
        #gray2_rgb = tools_draw_numpy.draw_circle(gray2_rgb, coord2[i, 1], coord2[i, 0], 4, color)
        color = numpy.ndarray.tolist(color[0,0])
        cv2.circle(gray1_rgb, (coord1[i, 0], coord1[i, 1]), 4, color,thickness=-1)
        cv2.circle(gray2_rgb, (coord2[i, 0], coord2[i, 1]), 4, color,thickness=-1)
    return gray1_rgb, gray2_rgb
# ---------------------------------------------------------------------------------------------------------------------


def get_best_matches(image1,image2,disp_v1, disp_v2, disp_h1, disp_h2, window_size=15,step=10):

    N = int(image1.shape[0] * image1.shape[1] / step / step)
    rand_points = numpy.random.rand(N,2)
    rand_points[:, 0] = window_size + (rand_points[:, 0]*(image1.shape[0]-2*window_size))
    rand_points[:, 1] = window_size + (rand_points[:, 1]*(image1.shape[1]-2*window_size))

    coord1,coord2,quality = [],[],[]

    for each in rand_points:
        row,col = int(each[0]),int(each[1])

        template = tools_image.crop_image(image1,row-window_size,col-window_size,row+window_size,col+window_size)
        top,left,bottom,right = row - window_size + disp_v1, col - window_size + disp_h1,row + window_size + disp_v2,col + window_size + disp_h2
        field = tools_image.crop_image(image2,top,left,bottom,right)

        if numpy.min(template[:,:,0])==numpy.max(template[:,:,0]) and numpy.min(template[:, :, 1]) == numpy.max(template[:, :, 1]) and numpy.min(template[:, :, 2]) == numpy.max(template[:, :, 2]):
            q = numpy.full((disp_v2-disp_v1,disp_h2-disp_h1),-1)
        else:
            q = cv2.matchTemplate(field, template, method=cv2.TM_CCOEFF_NORMED)
            q = q[1:, 1:]
            q = (q + 1) * 128

        idx = numpy.argmax(q.flatten())
        q_best = numpy.max(q.flatten())
        qq=q[int(idx/q.shape[1]), idx % q.shape[1]]

        if q_best>0:
            dr = int(idx/q.shape[1])+disp_v1
            dc = idx % q.shape[1]+disp_h1

            if col + dc>=0 and col + dc<image1.shape[1] and row + dr>=0 and row + dr<image1.shape[0]:
                coord1.append([col    ,row    ])
                coord2.append([col+ dc, row+dr])
                quality.append(q_best)

    return numpy.array(coord1), numpy.array(coord2),numpy.array(quality)

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

    desc = None

    kp, desc = detector.detectAndCompute(im1_gray, None)

    points = None
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
def get_disparity_SGBM(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2):
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
def get_disparity_BM(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2):
    max = numpy.maximum(math.fabs(disp_h1), math.fabs(disp_h2))
    levels = int(1 + (max) / 16) * 16
    stereo = cv2.StereoBM_create(numDisparities=levels, blockSize=15)
    displ = stereo.compute(imgL, imgR)

    dispr = numpy.flip(stereo.compute(numpy.flip(imgR, axis=1), numpy.flip(imgL, axis=1)), axis=1)
    return -displ / 16, -dispr / 16

# ---------------------------------------------------------------------------------------------------------------------
def get_disparity_from_matches(rows,cols, coord1, coord2,disp_v1, disp_v2, disp_h1, disp_h2):

    value = []
    for i in range(0, coord1.shape[0]):
        dx =  255-255*(coord2[i,0]-coord1[i,0]-disp_h1)/(disp_h2-disp_h1)
        value.append(dx)
    value = numpy.array(value)

    displ = interpolate_image_by_matches(rows, cols, coord1, value)
    dispr = interpolate_image_by_matches(rows, cols, coord2, value)

    displ_jet = tools_image.hitmap2d_to_jet(displ)
    dispr_jet = tools_image.hitmap2d_to_jet(dispr)

    return displ_jet, dispr_jet

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
                distances.extend(dist)

    all_matches_scr = numpy.array(all_matches_scr).astype(numpy.int)
    all_matches_dst = numpy.array(all_matches_dst).astype(numpy.int)
    distances = numpy.array(distances).flatten()


    return all_matches_scr[1:], all_matches_dst[1:], distances
# ---------------------------------------------------------------------------------------------------------------------
def get_matches_from_keypoints_desc(points_source,des_source, points_destin,des_destin,matchtype='knn'):
    src, dst, distance = [], [], []

    if points_source is None or des_source is None or des_source is None or des_destin is None:
        return None, None, None

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


    for m in matches:
        if m.queryIdx < points_destin.shape[0] and m.trainIdx < points_source.shape[0]:
            src.append(points_source[m.trainIdx])
            dst.append(points_destin[m.queryIdx])
            distance.append(m.distance)

    return numpy.array(src), numpy.array(dst), numpy.array(distance)
# ---------------------------------------------------------------------------------------------------------------------
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
def reproject_matches(image1, image2, crd1, crd2, window_size=20, fill_declines=False, background_color=[128, 128, 128]):

    R=window_size
    canv1 = tools_image.canvas_extrapolate(image1, image1.shape[0]+2*R, image1.shape[1]+2*R)
    canv2 = tools_image.canvas_extrapolate(image2, image2.shape[0]+2*R, image2.shape[1]+2*R)
    coord1 = crd1.copy()
    coord2 = crd2.copy()

    acc1_color = numpy.zeros((canv1.shape[0],canv1.shape[1],3))
    acc1_count = numpy.zeros((canv1.shape[0],canv1.shape[1]))
    acc2_color = numpy.zeros((acc1_color.shape))
    acc2_count = numpy.zeros((canv1.shape[0],canv1.shape[1]))

    if coord1.size == 0:
        mask1 = acc1_count[R:-R, R:-R].astype(numpy.uint8)
        mask2 = acc2_count[R:-R, R:-R].astype(numpy.uint8)
        acc1_color = acc1_color[R:-R, R:-R].astype(numpy.uint8)
        acc2_color = acc2_color[R:-R, R:-R].astype(numpy.uint8)
        acc1_color[:,:] = background_color
        acc2_color[:, :] = background_color
        return acc1_color, acc2_color, mask1, mask2

    coord1[:,0] += R
    coord1[:,1] += R
    coord2[:,0] += R
    coord2[:,1] += R

    # declines
    if fill_declines==True:
        acc1_color[R:-R, R:-R] = image1[:,:]
        acc1_count[R:-R, R:-R] = 1
        acc2_color[R:-R, R:-R] = image2[:,:]
        acc2_count[R:-R, R:-R] = 1

    for i in range (0,coord1.shape[0]):
        small = canv1[coord1[i,1]-R:coord1[i,1]+R,coord1[i,0]-R:coord1[i,0]+R]
        acc1_color   [coord2[i,1]-R:coord2[i,1]+R,coord2[i,0]-R:coord2[i,0]+R] +=small
        acc1_count   [coord2[i,1]-R:coord2[i,1]+R,coord2[i,0]-R:coord2[i,0]+R] += 1

        small = canv2[coord2[i, 1] - R:coord2[i, 1] + R, coord2[i, 0] - R:coord2[i, 0] + R]
        acc2_color[coord1[i, 1] - R:coord1[i, 1] + R, coord1[i, 0] - R:coord1[i, 0] + R] += small
        acc2_count[coord1[i, 1] - R:coord1[i, 1] + R, coord1[i, 0] - R:coord1[i, 0] + R] += 1


    idx = numpy.where(acc1_count > 0)
    acc1_color[idx[0], idx[1], 0] /= acc1_count[idx]
    acc1_color[idx[0], idx[1], 1] /= acc1_count[idx]
    acc1_color[idx[0], idx[1], 2] /= acc1_count[idx]
    idx = numpy.where(acc1_count == 0)
    acc1_color[idx[0], idx[1], :] = background_color
    acc1_color = acc1_color[R:-R, R:-R].astype(numpy.uint8)


    idx = numpy.where(acc2_count > 0)
    acc2_color[idx[0], idx[1], 0] /= acc2_count[idx]
    acc2_color[idx[0], idx[1], 1] /= acc2_count[idx]
    acc2_color[idx[0], idx[1], 2] /= acc2_count[idx]
    idx = numpy.where(acc2_count == 0)
    acc2_color[idx[0], idx[1], :] = background_color
    acc2_color = acc2_color[R:-R, R:-R].astype(numpy.uint8)

    mask1 = acc1_count[R:-R, R:-R].astype(numpy.uint8)
    mask2 = acc2_count[R:-R, R:-R].astype(numpy.uint8)

    return acc1_color,acc2_color,mask1,mask2
# ---------------------------------------------------------------------------------------------------------------------
def interpolate_image_by_matches(rows,cols,coord1, value):

    if coord1.shape[0]<=2:
        return result

    x = coord1[:,0]
    y = coord1[:,1]
    z = value

    xi, yi = numpy.meshgrid(numpy.arange(0, cols, 1), numpy.arange(0, rows, 1))
    zi = griddata((x, y), z, (xi, yi), method='linear')




    return zi
# ---------------------------------------------------------------------------------------------------------------------

