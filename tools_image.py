import cv2
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------------------------
def canvas_extrapolate_gray(gray, new_height, new_width):

    newimage = numpy.zeros((new_height,new_width),numpy.uint8)
    shift_x = int((newimage.shape[0] - gray.shape[0]) / 2)
    shift_y = int((newimage.shape[1] - gray.shape[1]) / 2)
    newimage[shift_x:shift_x + gray.shape[0], shift_y:shift_y + gray.shape[1]] = gray[:, :]

    newimage[:shift_x, shift_y:shift_y + gray.shape[1]] = gray[0, :]
    newimage[shift_x + gray.shape[0]:, shift_y:shift_y + gray.shape[1]] = gray[-1, :]

    for row in range(0, newimage.shape[0]):
        newimage[row, :shift_y] = newimage[row, shift_y]
        newimage[row, shift_y + gray.shape[1]:] = newimage[row, shift_y + gray.shape[1] - 1]

    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def canvas_extrapolate(img,new_height,new_width):

    newimage = numpy.zeros((new_height,new_width,3),numpy.uint8)
    shift_x = int((newimage.shape[0] - img.shape[0]) / 2)
    shift_y = int((newimage.shape[1] - img.shape[1]) / 2)
    newimage[shift_x:shift_x + img.shape[0], shift_y:shift_y + img.shape[1],:] = img[:,:,:]

    newimage[:shift_x, shift_y:shift_y + img.shape[1],:] = img[0, :,:]
    newimage[shift_x + img.shape[0]:, shift_y:shift_y + img.shape[1],:] = img[-1, :,:]

    for row in range(0, newimage.shape[0]):
        newimage[row, :shift_y] = newimage[row, shift_y]
        newimage[row, shift_y + img.shape[1]:] = newimage[row, shift_y + img.shape[1] - 1]

    return newimage
# ---------------------------------------------------------------------------------------------------------------------
def crop_image(img, top, left, bottom, right,extrapolate_border=False):

    if top >=0 and left >= 0 and bottom <= img.shape[0] and right <= img.shape[1]:
        return img[top:bottom, left:right]
    if len(img.shape)==2:
        result = numpy.zeros((bottom-top,right-left),numpy.uint8)
    else:
        result = numpy.zeros((bottom - top, right - left,3),numpy.uint8)

    if left > img.shape[1] or right <= 0 or top > img.shape[0] or bottom <= 0:
        return result

    if top < 0:
        start_row_source = 0
        start_row_target = - top
    else:
        start_row_source = top
        start_row_target = 0

    if bottom > img.shape[0]:
        finish_row_source = img.shape[0]
        finish_row_target = img.shape[0]-top
    else:
        finish_row_source = bottom
        finish_row_target = bottom-top

    if left < 0:
        start_col_source = 0
        start_col_target = - left
    else:
        start_col_source = left
        start_col_target = 0

    if right > img.shape[1]:
        finish_col_source = img.shape[1]
        finish_col_target = img.shape[1]-left
    else:
        finish_col_source = right
        finish_col_target = right-left

    result[start_row_target:finish_row_target,start_col_target:finish_col_target] = img[start_row_source:finish_row_source,start_col_source:finish_col_source]

    if extrapolate_border == True:
        fill_border(result,start_row_target,start_col_target,finish_row_target,finish_col_target)

    return result
# ---------------------------------------------------------------------------------------------------------------------
def get_mask(image_layer,background_color=(0,0,0)):

    mask_layer = numpy.zeros((image_layer.shape[0], image_layer.shape[1]), numpy.uint8)
    mask_layer[numpy.where(image_layer[:, :, 0] == background_color[0])] += 1
    mask_layer[numpy.where(image_layer[:, :, 1] == background_color[1])] += 1
    mask_layer[numpy.where(image_layer[:, :, 2] == background_color[2])] += 1
    mask_layer[mask_layer != 3] = 255
    mask_layer[mask_layer == 3] = 0

    return mask_layer
# ---------------------------------------------------------------------------------------------------------------------
def put_layer_on_image(image_background,image_layer,background_color=(0,0,0)):

    mask_layer = numpy.zeros((image_layer.shape[0],image_layer.shape[1]),numpy.uint8)
    mask_layer[numpy.where(image_layer[:, :, 0] == background_color[0])] += 1
    mask_layer[numpy.where(image_layer[:, :, 1] == background_color[1])] += 1
    mask_layer[numpy.where(image_layer[:, :, 2] == background_color[2])] += 1
    mask_layer[mask_layer !=3 ] = 255
    mask_layer[mask_layer == 3] = 0

    mask_layer_inv = cv2.bitwise_not(mask_layer)

    img1 = cv2.bitwise_and(image_background, image_background, mask=mask_layer_inv).astype(numpy.uint8)
    img2 = cv2.bitwise_and(image_layer     , image_layer     , mask=mask_layer).astype(numpy.uint8)


    im_result = cv2.add(img1, img2).astype(numpy.uint8)

    return im_result
#--------------------------------------------------------------------------------------------------------------------------
def capture_image_to_disk(out_filename):

    cap = cv2.VideoCapture(0)

    while (True):

        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(out_filename,frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return
#--------------------------------------------------------------------------------------------------------------------------
def desaturate_2d(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#--------------------------------------------------------------------------------------------------------------------------
def desaturate(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = numpy.zeros((image.shape[0], image.shape[1], 3)).astype(numpy.uint8)
    result[:, :, 0] = gray[:, :]
    result[:, :, 1] = gray[:, :]
    result[:, :, 2] = gray[:, :]
    return result
#--------------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_jet(hitmap_2d):
    colormap = cv2.COLORMAP_JET
    hitmap_RGB_jet = cv2.applyColorMap(hitmap_2d.astype(numpy.uint8), colormap)
    return hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def hsv2bgr(hsv):
    return cv2.cvtColor(numpy.array([hsv[0], hsv[1], hsv[2]], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)
# ----------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_viridis(hitmap_2d):
    colormap = (numpy.array(cm.cmaps_listed['viridis'].colors)*256).astype(int)
    colormap = numpy.flip(colormap,axis=1)

    colormap2 = plt.get_cmap('RdBu')

    hitmap_RGB_gre  = numpy.zeros((hitmap_2d.shape[0],hitmap_2d.shape[1],3)).astype(numpy.uint8)
    hitmap_RGB_gre[:, :, 0] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 1] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 2] = hitmap_2d[:, :]
    hitmap_RGB_jet = hitmap_RGB_gre.copy()

    for i in range (0,hitmap_RGB_jet.shape[0]):
        for j in range(0, hitmap_RGB_jet.shape[1]):
            c= int(hitmap_RGB_gre[i, j, 0])
            hitmap_RGB_jet[i,j] = colormap[c]

    return hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def fill_border(array,row_top,col_left,row_bottom,col_right):
    array[:row_top,:]=array[row_top,:]
    array[row_bottom:, :] = array[row_bottom-1, :]

    array = cv2.transpose(array)
    #array=array.T
    array[:col_left,:]=array[col_left,:]
    array[col_right:, :] = array[col_right - 1, :]
    #array = array.T
    array = cv2.transpose(array)
    return array
# ----------------------------------------------------------------------------------------------------------------------
def shift_image_vert(image,dv):
    res = image.copy()

    dv = -dv

    if dv != 0:
        if (dv > 0):
            res[0:image.shape[0] - dv, :] = image[dv:image.shape[0], :]
            res[image.shape[0] - dv:, :] = image[image.shape[0] - 1, :]
        else:
            res[-dv:image.shape[0], :] = image[0:image.shape[0] + dv, :]
            res[:-dv, :, :] = image[0, :]



    return res
# ----------------------------------------------------------------------------------------------------------------------
def shift_image(image,dv,dh):

    res = image.copy()

    if dv!=0:
        res = shift_image_vert(image, dv)

    res2 = cv2.transpose(image)
    res2 = shift_image_vert(res2, dh)
    res2 = cv2.transpose(res2)

    return res2
# ----------------------------------------------------------------------------------------------------------------------
def blend_multi_band(left, rght, background_color=(255, 255, 255)):
    def GaussianPyramid(img, leveln):
        GP = [img]
        for i in range(leveln - 1):
            GP.append(cv2.pyrDown(GP[i]))
        return GP
    # --------------------------------------------------------------------------------------------------------------------------
    def LaplacianPyramid(img, leveln):
        LP = []
        for i in range(leveln - 1):
            next_img = cv2.pyrDown(img)
            size = img.shape[1::-1]

            temp_image = cv2.pyrUp(next_img, dstsize=img.shape[1::-1])
            LP.append(img - temp_image)
            img = next_img
        LP.append(img)
        return LP
    # --------------------------------------------------------------------------------------------------------------------------
    def blend_pyramid(LPA, LPB, MP):
        blended = []
        for i, M in enumerate(MP):
            blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
        return blended

    # --------------------------------------------------------------------------------------------------------------------------
    def reconstruct_from_pyramid(LS):
        img = LS[-1]
        for lev_img in LS[-2::-1]:
            img = cv2.pyrUp(img, dstsize=lev_img.shape[1::-1])
            img += lev_img
        return img

    # --------------------------------------------------------------------------------------------------------------------------
    def get_borders(image, bg=(255, 255, 255)):

        if (bg == (255, 255, 255)):
            prj = numpy.min(image, axis=0)
        else:
            prj = numpy.max(image, axis=0)

        flg = (prj == bg)[:, 0]

        l = numpy.argmax(flg == False)
        r = numpy.argmin(flg == False)
        return l, r, 0, 0
    # --------------------------------------------------------------------------------------------------------------------------
    left_l, left_r, left_t, left_b = get_borders(left, background_color)
    rght_l, rght_r, rght_t, rght_b = get_borders(rght, background_color)
    border = int((left_r+rght_l)/2)

    mask = numpy.zeros(left.shape)
    mask[:, :border] = 1

    leveln = int(numpy.floor(numpy.log2(min(left.shape[0], left.shape[1]))))

    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(numpy.array(left).astype('float'), leveln)
    LPB = LaplacianPyramid(numpy.array(rght).astype('float'), leveln)
    blended = blend_pyramid(LPA, LPB, MP)

    result = reconstruct_from_pyramid(blended)
    result[result > 255] = 255
    result[result < 0] = 0
    return result
#----------------------------------------------------------------------------------------------------------------------
def blend_avg(img1, img2,background_color=(255,255,255),weight=0.5):

    im1 = put_layer_on_image(img1, img2,background_color)
    im2 = put_layer_on_image(img2, img1,background_color)

    res = cv2.add(im1*(1-weight), im2*(weight))

    return res.astype(numpy.uint8)
#----------------------------------------------------------------------------------------------------------------------
