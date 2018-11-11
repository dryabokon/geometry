import cv2
import numpy
from PIL import Image
import matplotlib.cm as cm
#--------------------------------------------------------------------------------------------------------------------------
def cut_PIL_Image(img,l,t,w,h,target_width,target_height):

    width = int(w * 3)
    cx = l+w/2
    cy = t+h/2

    left = int(cx - width / 2.0)
    right  = int(cx + width / 2.0)
    top    = int(cy - 9 * width / 10.0)
    bottom = int(cy + 1 * width / 10.0)


    if(left>=0 and right < img.width and top >=0 and bottom < img.height):
        cropped = img.crop((left, top, right, bottom))
    else:
        if(top < 0):
            cropped = Image.new(mode=img.mode, size=(right-left, right-left))
            cropped.paste(img.crop((left, 0, right, bottom)),(0,-top))
            cropped.paste(img.crop((left,0,right,1)).resize((cropped.width,-top)),(0,0)) #cropped.save('C:/Users/dryabokon/source/Digits/cars/Ex42/audi_100/crop.jpg')
            if (left < 0):
                temp = cropped.crop((-left, 0, -left+1,cropped.height)).resize((-left, cropped.height))
                cropped.paste(temp, (0, 0))

            if (right >= img.width):
                temp = cropped.crop((cropped.width-(right-img.width)-1,0,cropped.width-(right-img.width),cropped.height))
                temp = temp.resize((right-img.width+1,cropped.height))
                cropped.paste(temp,(cropped.width-(right-img.width+1),0))

        elif (bottom >= img.height):
            cropped = Image.new(mode=img.mode, size=(right - left, right - left))
            cropped.paste(img.crop((left, top, right, img.height)), (0, 0))
            temp = img.crop((left, img.height - 1, right, img.height))
            temp.resize((cropped.width, bottom - img.height+1))
            cropped.paste(temp, (0,cropped.height-(bottom-img.height+1)))
            if (left < 0):
                temp = cropped.crop((-left, 0, -left+1,cropped.height)).resize((-left, cropped.height))
                cropped.paste(temp, (0, 0))

            if (right >= img.width):
                temp = cropped.crop((cropped.width-(right-img.width)-1,0,cropped.width-(right-img.width),cropped.height))
                temp = temp.resize((right-img.width+1,cropped.height))
                cropped.paste(temp,(cropped.width-(right-img.width+1),0))

        elif (left < 0):
            cropped = Image.new(mode=img.mode, size=(bottom - top, bottom - top))
            cropped.paste(img.crop((0, top, right, bottom)), (-left, 0))
            cropped.paste(img.crop((0,top, 1, bottom)).resize((-left,cropped.height)), (0, 0))

        elif (right >= img.width):
            cropped = Image.new(mode=img.mode, size=(bottom - top, bottom - top))
            cropped.paste(img.crop((left, top, img.width, bottom)), (0, 0))
            cropped.paste(img.crop((img.width-1,top, img.width, bottom)).resize((right-img.width+1,cropped.height)), (cropped.width-(right-img.width+1),0))

    res = cropped.resize((target_width, target_height))

    return res
#--------------------------------------------------------------------------------------------------------------------------
def crop_image(img, bbox):

    def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2

    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        temp_img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return temp_img[y1:y2, x1:x2, :]
    return img[y1:y2, x1:x2, :]
#--------------------------------------------------------------------------------------------------------------------------
def put_layer_on_image(image_background,image_layer):

    ret, mask = cv2.threshold(image_layer, 0, 255, cv2.THRESH_BINARY)

    mask = mask[:,:,0]
    mask_inv = cv2.bitwise_not(mask)

    img1 = cv2.bitwise_and(image_background, image_background, mask=mask_inv)
    img2 = cv2.bitwise_and(image_layer     , image_layer     , mask=mask)

    im_result = cv2.add(img1, img2)

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

    hitmap_RGB_gre = numpy.zeros((hitmap_2d.shape[0], hitmap_2d.shape[1], 3)).astype(numpy.uint8)
    hitmap_RGB_gre[:, :, 0] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 1] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 2] = hitmap_2d[:, :]
    hitmap_RGB_jet = cv2.applyColorMap(255 - hitmap_2d, colormap)
    return hitmap_RGB_gre, hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def hitmap2d_to_viridis(hitmap_2d):
    colormap = (numpy.array(cm.cmaps_listed['viridis'].colors)*256).astype(int)

    hitmap_RGB_gre  = numpy.zeros((hitmap_2d.shape[0],hitmap_2d.shape[1],3)).astype(numpy.uint8)
    hitmap_RGB_gre[:, :, 0] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 1] = hitmap_2d[:, :]
    hitmap_RGB_gre[:, :, 2] = hitmap_2d[:, :]
    hitmap_RGB_jet = hitmap_RGB_gre.copy()

    for i in range (0,hitmap_RGB_jet.shape[0]):
        for j in range(0, hitmap_RGB_jet.shape[1]):
            c= int(hitmap_RGB_gre[i, j, 0])
            hitmap_RGB_jet[i,j] = colormap[c]

    return hitmap_RGB_gre, hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def fill_border(array,row_top,col_left,row_bottom,col_right):
    array[:row_top,:]=array[row_top,:]
    array[row_bottom:, :] = array[row_bottom-1, :]

    array=array.T
    array[:col_left,:]=array[col_left,:]
    array[col_right:, :] = array[col_right - 1, :]
    array = array.T
    return array
# ----------------------------------------------------------------------------------------------------------------------
def shift_image(gray,dv,dh):
    res = gray.copy()

    dv,dh=-dv,-dh

    if (dv>0):
        res[0:gray.shape[0]-dv , :] = gray[dv:gray.shape[0]  , :]
        res[  gray.shape[0]-dv:, :] = gray[   gray.shape[0]-1, :]
    else:
        res[-dv:gray.shape[0], :] = gray[0:gray.shape[0]+dv, :]
        res[   :-dv          , :] = gray[0                 , :]

    res = res.T
    res2 = res.copy()

    if (dh>0):
        res2[0:gray.shape[1]-dh, :] = res[dh:gray.shape[1]  ,:]
        res2[  gray.shape[1]-dh:,:] = res[   gray.shape[1]-1,:]
    else:
        res2[-dh:gray.shape[1]  ,:] = res[0:gray.shape[1]+dh,:]
        res2[   :-dh            ,:] = res[0, :]

    res2 = res2.T

    return res2
# ----------------------------------------------------------------------------------------------------------------------