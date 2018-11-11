#----------------------------------------------------------------------------------------------------------------------
import os
import numpy
import cv2
from scipy.misc import toimage
from scipy import ndimage,interpolate
from scipy.signal import gaussian, fftconvolve
#----------------------------------------------------------------------------------------------------------------------
import tools_IO as IO
import tools_image
import tools_draw_numpy
import tools_calibrate
#----------------------------------------------------------------------------------------------------------------------
def calc_hit_field(image_base, image_pattern):

    x1 = int(1*image_pattern.shape[0]/4)
    x2 = int(3*image_pattern.shape[0]/4)
    y1 = int(1*image_pattern.shape[1]/4)
    y2 = int(3*image_pattern.shape[1]/4)

    hitmap1 = calc_hit_field_basic(image_base, image_pattern             ).astype(numpy.float)
    hitmap2 = calc_hit_field_basic(image_base, image_pattern[x1:x2,y1:y2]).astype(numpy.float)
    hitmap3 = calc_hit_field_basic(image_base, image_pattern[x1:x2,  :  ]).astype(numpy.float)
    hitmap4 = calc_hit_field_basic(image_base, image_pattern[:    ,y1:y2]).astype(numpy.float)


    hitmap=(hitmap4.copy()).astype(numpy.float)
    hitmap=hitmap1*hitmap2*hitmap3*hitmap4

    N=int(image_pattern.shape[0]*0.25)
    hitmap = ndimage.convolve(hitmap, numpy.ones((N,N)), mode='constant')
    max = numpy.max(hitmap)/255
    hitmap/= max

    return hitmap.astype(numpy.uint8)
#----------------------------------------------------------------------------------------------------------------------
def calc_hit_field_basic(image_base, image_pattern,mask=None):

    hitmap_gray = cv2.matchTemplate(image_base, image_pattern, method=cv2.TM_CCOEFF_NORMED)

    min = hitmap_gray.min()
    max = hitmap_gray.max()
    hitmap_gray = (225 * (hitmap_gray - min) / (max - min)).astype(numpy.uint8)
    hitmap_gray = numpy.reshape(hitmap_gray,(hitmap_gray.shape[0],hitmap_gray.shape[1],1))

    hitmap_gray2 = numpy.zeros((image_base.shape[0],image_base.shape[1],1)).astype(numpy.uint8)
    shift_x = int((image_base.shape[0] - hitmap_gray.shape[0])/2)
    shift_y = int((image_base.shape[1] - hitmap_gray.shape[1])/2)
    hitmap_gray2[shift_x:shift_x+hitmap_gray.shape[0], shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[: , :, 0]


    hitmap_gray2[:shift_x                            , shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[0 , :, 0]
    hitmap_gray2[shift_x + hitmap_gray.shape[0]:     , shift_y:shift_y + hitmap_gray.shape[1], 0] = hitmap_gray[-1, :, 0]

    for row in range(0,hitmap_gray2.shape[0]):
        hitmap_gray2[row,:shift_y                        , 0] = hitmap_gray2[row, shift_y                          , 0]
        hitmap_gray2[row, shift_y + hitmap_gray.shape[1]:, 0] = hitmap_gray2[row, shift_y + hitmap_gray.shape[1]-1 , 0]


    hitmap_2d       = numpy.reshape(hitmap_gray2, (hitmap_gray2.shape[0], hitmap_gray2.shape[1]))

    return hitmap_2d
#----------------------------------------------------------------------------------------------------------------------
def rolling_std(gray,drow,dcol):

    g1_summ = numpy.cumsum(numpy.cumsum(gray, axis=0), axis=1)
    g2_summ = numpy.cumsum(numpy.cumsum(numpy.multiply(gray, gray), axis=0), axis=1)

    weights = numpy.zeros(((2+2*drow), (2+2*dcol)))
    weights[ 0, 0] = +1
    weights[-1,-1] = +1
    weights[ 0,-1] = -1
    weights[-1, 0] = -1

    summ  = ndimage.convolve(g1_summ, weights, mode='constant', origin=((-1, -1)))
    summ2 = ndimage.convolve(g2_summ, weights, mode='constant', origin=((-1, -1)))
    summ [: , gray.shape[1]-dcol:] = 0
    summ [gray.shape[0] - drow:,:] = 0
    summ2[: , gray.shape[1]-dcol:] = 0
    summ2[gray.shape[0] - drow:,:] = 0


    norm = ((1 + 2 * drow) * (1 + 2 * dcol))
    res = summ2/norm-(summ/norm)*(summ/norm)
    res = numpy.sqrt(res)

    res=tools_image.fill_border(res,drow,dcol,res.shape[0]-drow,res.shape[1]-dcol)

    return res
# ----------------------------------------------------------------------------------------------------------------------
def save_quality_to_disk(result_4d,disp_v1,disp_v2,disp_h1,disp_h2,path_debug):

    if path_debug!='' and (not os.path.exists(path_debug)):
        os.makedirs(path_debug)
    else:
        IO.remove_files(path_debug)

    ######
    result_3d = numpy.min(result_4d, axis=2)
    IO.remove_files(path_debug)

    for dh in range(disp_h1,disp_h2):
        hitmap_2d = result_3d[:,:,dh-disp_h1].astype(numpy.uint8)
        hitmap_RGB_gre, hitmap_RGB_jet = tools_image.hitmap2d_to_jet(hitmap_2d)
        toimage(hitmap_RGB_jet).save(path_debug+ ('q1_%03d.bmp' % (dh) ))

    ######

    result_3d = numpy.min(result_4d, axis=2)
    for row in range(0,result_4d.shape[0]):
        hitmap_2d = result_3d[row, :, :].astype(numpy.uint8)
        hitmap_RGB_gre, hitmap_RGB_jet = tools_image.hitmap2d_to_jet(hitmap_2d.T)
        toimage(hitmap_RGB_jet).save(path_debug + ('q2_%03d.bmp' % (row)))

    return
# ----------------------------------------------------------------------------------------------------------------------
def calc_match_quality(image1,image2,disp_v1,disp_v2,disp_h1,disp_h2,window_size=5,path_debug=''):

    if len(image1.shape)!=2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if len(image2.shape)!=2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    result_4d = numpy.full((image1.shape[0], image1.shape[1],disp_v2-disp_v1,disp_h2 - disp_h1),-1)

    for dh in range(disp_h1,disp_h2):
        if dh==-15:
            dh=dh
        for dv in range(disp_v1,disp_v2):
            gray2_shifted = tools_image.shift_image(image2,-dv,-dh)
            gray = 127+(image1-gray2_shifted)/2

            hitmap_2d = rolling_std(gray, window_size, window_size)
            result_4d[:,:,dv-disp_v1,dh-disp_h1]=hitmap_2d

    min=numpy.min(result_4d)
    result_4d-=min
    max= numpy.max(result_4d)
    result_4d=255-(255*result_4d/max).astype(numpy.uint8)

    if path_debug!='':
        save_quality_to_disk(result_4d,disp_v1,disp_v2,disp_h1,disp_h2,path_debug)

    return result_4d
# ----------------------------------------------------------------------------------------------------------------------
def visualize_matched_pairs_hor_v1(image1, image2, coord):

    result_image1 = numpy.zeros((image1.shape[0]*2,image1.shape[1],3),numpy.uint8)
    result_image1[:image1.shape[0],:,:]   = image1[:, :, :]
    result_image1[image1.shape[0]:, :, :] = image2[:, :, :]

    for i in range (0,coord.shape[0]):
        row1, col1 =  coord[i,0]                , coord[i,1]
        row2, col2 =  coord[i,2]+image1.shape[0], coord[i,3]
        result_image1 = tools_draw_numpy.draw_circle(result_image1, row1, col1, 3,[0, 0, 200])
        result_image1 = tools_draw_numpy.draw_circle(result_image1, row2, col2, 3,[0, 0, 200])
        result_image1 = tools_draw_numpy.draw_line  (result_image1, row1, col1, row2, col2, [0, 100, 200],0.75)

    return result_image1
# ----------------------------------------------------------------------------------------------------------------------
def visualize_matched_pairs_ver_v1(image1, image2, coord):
    result_image2 = numpy.zeros((image1.shape[0], image1.shape[1] * 2, 3), numpy.uint8)
    result_image2[:, :image1.shape[1], :] = image1[:, :, :]
    result_image2[:, image1.shape[1]:, :] = image2[:, :, :]

    for i in range(0, coord.shape[0]):
        row1, col1 = coord[i, 0], coord[i, 1]
        row2, col2 = coord[i, 2], coord[i, 3] + image1.shape[1]
        result_image2 = tools_draw_numpy.draw_circle(result_image2, row1, col1, 3, [0, 0, 200])
        result_image2 = tools_draw_numpy.draw_circle(result_image2, row2, col2, 3, [0, 0, 200])
        result_image2 = tools_draw_numpy.draw_line(result_image2, row1, col1, row2, col2, [0, 100, 200], 0.75)

    return result_image2
# ----------------------------------------------------------------------------------------------------------------------
def visualize_matched_pairs_v2(image1, coord):

    result_image1     = image1.copy()
    result_image1_gre = numpy.zeros((image1.shape[0], image1.shape[1], 3)).astype(numpy.uint8)
    result_image1_gre[:, :, 0] = image1[:, :, 0]
    result_image1_gre[:, :, 1] = image1[:, :, 0]
    result_image1_gre[:, :, 2] = image1[:, :, 0]

    for i in range (0,coord.shape[0]):
        row1, col1 =  coord[i,0], coord[i,1]
        row2, col2 =  coord[i,2], coord[i,3]
        result_image1_gre = tools_draw_numpy.draw_circle(result_image1_gre, row1, col1, 3,[0, 0, 200])
        result_image1_gre = tools_draw_numpy.draw_circle(result_image1_gre, row2, col2, 3,[0, 0, 200])
        result_image1_gre = tools_draw_numpy.draw_line  (result_image1_gre, row1, col1, row2, col2, [0, 100, 200],0.75)

    return result_image1_gre
# ----------------------------------------------------------------------------------------------------------------------
def visualize_matches_map(image, disp_row,disp_col,disp_v1, disp_v2, disp_h1, disp_h2):
    res=image.copy()
    image_disp_hor = numpy.zeros((image.shape[0],image.shape[1])).astype(numpy.float32)
    image_disp_hor [:,:]=disp_col[:,:]

    for row in range(0,image.shape[0]):
        for col in range(0, image.shape[1]):
            r = row + disp_row[row, col]
            c = col + disp_col[row, col]
            r= numpy.maximum(0,numpy.minimum(image.shape[0]-1,r))
            c= numpy.maximum(0,numpy.minimum(image.shape[1]-1,c))
            res[row,col] =image[r,c]

    image_disp_hor=(image_disp_hor-disp_h1)*255/(disp_h2-disp_h1)
    image_disp_hor_RGB_gre, image_disp_hor_RGB_jet = tools_image.hitmap2d_to_jet(image_disp_hor.astype(numpy.uint8))

    image_disp_hor_RGB_jet[disp_col>=disp_h2+1]=[128,128,128]



    return res,image_disp_hor_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def mse(im1_gray, im2_gray):

    if len(im1_gray.shape)!=2:
        im1_gray = cv2.cvtColor(im1_gray, cv2.COLOR_BGR2GRAY)

    if len(im2_gray.shape)!=2:
        im2_gray = cv2.cvtColor(im2_gray, cv2.COLOR_BGR2GRAY)

    a=numpy.zeros(im1_gray.shape).astype(numpy.float32)
    a+=im2_gray
    a-=im1_gray
    avg = numpy.average(a)
    a-=avg

    #d=numpy.sum((a)**2)
    #d/=float(im1_gray.shape[0] * im1_gray.shape[1])
    #d=math.sqrt(d)
    th = 64
    x = numpy.where(a>th)
    d2= 255.0*len(x[0])/(im1_gray.shape[0] * im1_gray.shape[1])
    return d2
# ----------------------------------------------------------------------------------------------------------------------
def ssim(im1, im2, k=(0.01, 0.03), l=255):

    if len(im1.shape)!=2:
        im1= cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    if len(im2.shape)!=2:
        im2= cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    win = numpy.array([gaussian(11, 1.5)])
    window = win * (win.T)
    """See https://ece.uwaterloo.ca/~z70wang/research/ssim/"""
    # Check if the window is smaller than the images.
    for a, b in zip(window.shape, im1.shape):
        if a > b:
            return None, None
    # Values in k must be positive according to the base implementation.
    for ki in k:
        if ki < 0:
            return None, None

    c1 = (k[0] * l) ** 2
    c2 = (k[1] * l) ** 2
    window = window/numpy.sum(window)

    mu1 = fftconvolve(im1, window, mode='valid')
    mu2 = fftconvolve(im2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = fftconvolve(im1 * im1, window, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(im2 * im2, window, mode='valid') - mu2_sq
    sigma12 = fftconvolve(im1 * im2, window, mode='valid') - mu1_mu2

    if c1 > 0 and c2 > 0:
        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den
    else:
        num1 = 2 * mu1_mu2 + c1
        num2 = 2 * sigma12 + c2
        den1 = mu1_sq + mu2_sq + c1
        den2 = sigma1_sq + sigma2_sq + c2
        ssim_map = numpy.ones(numpy.shape(mu1))
        index = (den1 * den2) > 0
        ssim_map[index] = (num1[index] * num2[index]) / (den1[index] * den2[index])
        index = (den1 != 0) & (den2 == 0)
        ssim_map[index] = num1[index] / den1[index]

    mssim = ssim_map.mean()
    return int(100*mssim)#, ssim_map
# ----------------------------------------------------------------------------------------------------------------------
def coordinates_to_images(image_base, coordinates, dx, dy):
    images=[]

    dx=int(dx)
    dy=int(dy)
    for i in range(0, coordinates.shape[0]):
        x = coordinates[i][0]
        y = coordinates[i][1]

        left = x - int(dx / 2)
        right = x - int(dx / 2) + dx
        top = y - int(dy / 2)
        bottom = y - int(dy / 2) + dy


        cut = tools_image.crop_image(image_base, (top,left,bottom,right))

        #cut = image_base[left:right, top:bottom]
        if (cut.shape[0], cut.shape[1]) == (dx, dy):
            images.append(cut)

    return numpy.array(images)
# ----------------------------------------------------------------------------------------------------------------------
def calc_matches_mat(quality_4d,disp_v1, disp_v2, disp_h1, disp_h2):

    quality_3d = numpy.max(quality_4d, axis=3)
    disp_row = numpy.argmax(quality_3d,axis=2)+disp_v1

    quality_3d = numpy.max(quality_4d, axis=2)
    disp_col = numpy.argmax(quality_3d, axis=2)+disp_h1

    return disp_row,disp_col
# ----------------------------------------------------------------------------------------------------------------------
def filter_outliers(image_base, coordinates,pattern,th=1.0,path_debug=''):

    if (coordinates.size == 0):
        return numpy.array([])

    if path_debug != '' and (not os.path.exists(path_debug)):
        os.makedirs(path_debug)
    else:
        IO.remove_files(path_debug)

    images = coordinates_to_images(image_base, coordinates, pattern.shape[0], pattern.shape[1])

    idx=[]
    for i in range(0, images.shape[0]):

        pattern    , im_aligned      = tools_calibrate.align_images(pattern,images[i],cv2.MOTION_AFFINE)
        value = mse(pattern, im_aligned)
        #value = ssim(pattern, im_aligned)
        if value<=th:
            idx.append(i)

        if path_debug != '':
            toimage(im_aligned).save(path_debug + '%03d_%03d.bmp' % (value,i))

    return coordinates[idx]
# ---------------------------------------------------------------------------------------------------------------------
def get_keypoints_desc(image1,detector='SIFT'):
    if len(image1.shape) == 2:
        im1_gray = im1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if detector == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    elif detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else: #detector == 'ORB'
        detector = cv2.ORB_create()

    kp, desc = detector.detectAndCompute(im1_gray, None)

    points=[]
    if (len(kp) > 0):
        points = numpy.array([kp[int(idx)].pt for idx in range(0, len(kp))]).astype(int)

    return points, desc
# ---------------------------------------------------------------------------------------------------------------------
def get_corners_Fast(image1):
    if len(image1.shape) == 2:
        im1_gray = im1.copy()
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
        im1_gray = im1.copy()
    else:
        im1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    detector = cv2.FastFeatureDetector_create()

    corners = cv2.goodFeaturesToTrack(im1_gray, 25, 0.01, 10)
    corners = corners.reshape(corners.shape[0], corners.shape[2]).astype(int)

    return corners
# ---------------------------------------------------------------------------------------------------------------------
def get_corners_Harris(image1):
    if len(image1.shape) == 2:
        im1_gray = im1.copy()
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
def get_matches_from_desc(d1, d2,normType=cv2.NORM_L1, crossCheck=True):
    matcher = cv2.BFMatcher(normType=normType, crossCheck=crossCheck)
    mtch = matcher.match(queryDescriptors=d1, trainDescriptors=d2)
    matches = numpy.array([(each.queryIdx, each.trainIdx) for each in mtch])
    return matches
# ---------------------------------------------------------------------------------------------------------------------
def get_matches_from_desc_limit_by_disp(k1,d1,k2,d2, disp_v1, disp_v2, disp_h1, disp_h2, normType=cv2.NORM_L1, crossCheck=True):
    matcher = cv2.BFMatcher(normType=normType, crossCheck=crossCheck)

    all_matches=numpy.zeros((1,2))
    distances = []
    for i in range (0,len(k1)):
        row1, col1 = k1[i].pt[1], k1[i].pt[0]
        desc2 = []
        idx = []
        for j in range(0, len(k2)):
            row2, col2 = k2[j].pt[1], k2[j].pt[0]
            if (col2 - col1 >= disp_h1) and (col2 - col1 < disp_h2) and (row2 - row1 >= disp_v1) and (row2 - row1 < disp_v2):
                desc2.append(d2[j])
                idx.append(j)

        desc2 = numpy.array(desc2)

        if desc2.size!=0:
            mtch = matcher.match(queryDescriptors=numpy.array([d1[i]]), trainDescriptors=desc2)
            if len(mtch)!=0:
                matches = numpy.array([(i, idx[each.trainIdx]) for each in mtch])
                dst = [each.distance for each in mtch]
                distances.append(dst)
                all_matches = numpy.vstack((all_matches,matches))

    all_matches = numpy.array(all_matches).astype(numpy.int)
    return all_matches[1:],numpy.array(distances)
# ---------------------------------------------------------------------------------------------------------------------
def calc_matches_pairs(image1,image2, disp_v1, disp_v2, disp_h1, disp_h2):
    dist_th = 2000000
    window_size = 25

    disp_col = numpy.full((image1.shape[0], image1.shape[1]), disp_h2+1, numpy.int)
    disp_row = numpy.full((image1.shape[0], image1.shape[1]), disp_v2+1, numpy.int)

    k1, d1, k2, d2 = get_keypoints_desc_SIFT(image1, image2)
    #matches = get_matches_from_desc(d1, d2)
    matches, dist  = get_matches_from_desc_limit_by_disp(k1,d1,k2, d2, disp_v1, disp_v2, disp_h1, disp_h2)

    coord=[]
    distance=[]

    for i in range (0,matches.shape[0]):
        each = matches[i]
        row1, col1 = k1[each[0]].pt[1], k1[each[0]].pt[0]
        row2, col2 = k2[each[1]].pt[1], k2[each[1]].pt[0]
        if (col2 - col1>=disp_h1) and (col2 - col1<disp_h2) and (row2 - row1>=disp_v1) and (row2 - row1 < disp_v2) and dist[i]<dist_th\
                and row1>window_size and row2>window_size and col1>window_size and col1>window_size \
                and image1.shape[0] - row1 > window_size and image1.shape[0] - row2 > window_size and image1.shape[1] - col1 > window_size and image1.shape[1] - col1 > window_size :
            coord.append((row1,col1,row2,col2))
            distance.append(dist[i])
            disp_col[int(row1), int(col1)], disp_row[int(row1), int(col1)] = int(col2 - col1), int(row2 - row1)

    coord = numpy.array(coord).astype(numpy.int)

    return coord,distance
# ---------------------------------------------------------------------------------------------------------------------
def E2E_generate_disp_map():

    path = '../_pal/ex03/'
    path_in1         = path + 'L.bmp'
    path_in2         = path + 'R.bmp'
    path_out1        = path + 'L_res.bmp'
    path_out2        = path + 'depth.bmp'
    path_out_debug   = path + 'debug/'
    window_size = 5
    disp_v1, disp_v2, disp_h1, disp_h2 = 0, 1, -70, -10

    image1 = cv2.imread(path_in1)
    image2 = cv2.imread(path_in2)


    quality_4d=calc_match_quality(image1,image2,disp_v1, disp_v2, disp_h1, disp_h2,window_size,path_out_debug)
    disp_row, disp_col = calc_matches_mat(quality_4d, disp_v1, disp_v2, disp_h1, disp_h2)
    res, image_disp_hor_RGB_jet = visualize_matches_map(image1, disp_row,disp_col,disp_v1, disp_v2, disp_h1, disp_h2)
    cv2.imwrite(path_out1, res)
    cv2.imwrite(path_out2,image_disp_hor_RGB_jet)

    return
# ---------------------------------------------------------------------------------------------------------------------
def E2E_generate_disp_pairs():
    #path = '../_pal/ex03/'
    #path_in1 = path + 'L.bmp'
    #path_in2 = path + 'R.bmp'

    path = '../_pal/ex01/'
    path_in1 = path + 'example.bmp'
    path_in2 = path + 'example.bmp'
    path_out1 = path + 'L_res.bmp'
    path_out2 = path + 'R_res.bmp'
    path_out_debug = path + 'debug/'
    window_size = 5

    image1 = cv2.imread(path_in1)
    image2 = cv2.imread(path_in2)

    disp_v1, disp_v2, disp_h1, disp_h2 = -10, +10, -170, -110
    pairs_hor,dist = calc_matches_pairs(image1,image2, disp_v1, disp_v2, disp_h1, disp_h2)
    image1_res  =  visualize_matched_pairs_hor(image1,image2, pairs_hor)
    cv2.imwrite(path_out1, image1_res)

    disp_v1, disp_v2, disp_h1, disp_h2 = -170, -100, -10, +10
    pairs_ver,dist = calc_matches_pairs(image1,image2, disp_v1, disp_v2, disp_h1, disp_h2)
    image2_res  =  visualize_matched_pairs_ver(image1,image2, pairs_ver)
    cv2.imwrite(path_out2, image2_res)

    return

# ---------------------------------------------------------------------------------------------------------------------