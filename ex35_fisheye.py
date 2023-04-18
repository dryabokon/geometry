import cv2
import numpy as numpy
from CV import tools_fisheye
from CV import tools_panoram
import tools_IO
import tools_image
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
C = tools_fisheye.Converter()
P = tools_panoram.Panoramer()
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_room_from_fisheye(image):

    im_equirect = C.fisheye2panoram(image, (512, 1024),y_min=0.5,y_max=1.0)
    cv2.imwrite(folder_out + 'im_equirect.png', im_equirect)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_room_planar_multiview(image):

    for i,offset in enumerate(numpy.arange(0,1024,8)):
        im = C.fisheye2panoram(image, (512, 1024), offset_x=offset,x_min=0,x_max=0.25,y_min=0.3,y_max=0.9)
        cv2.imwrite(folder_out + 'im_planar_%03d.png'%i,im)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_room_cubemap(image):

    rad_per_aper = 2.6
    aperture = image.shape[0] / 4

    im_equirect = C.fisheye2equirect(image, (512, 1024), aperture=aperture, radius=aperture*rad_per_aper,delx=0,dely=0)[:, ::-1]
    cv2.imwrite(folder_out + 'im_cube1.png', C.equirect2cubemap(im_equirect,side=256,modif=False,dice=True))
    cv2.imwrite(folder_out + 'im_cube2.png', C.equirect2cubemap(im_equirect,side=256,modif=True ,dice=True)[::-1,::-1])

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_rooms_multivew(folder_in,folder_out):

    offset0=-40

    filenames =tools_IO.get_filenames(folder_in, '*.jpg,*.png')
    step = float(1024/(len(filenames)-1))

    for i,filename in enumerate(filenames):
        offset = (offset0 + step * i + 1024) % 1024
        image = cv2.imread(folder_in + filename)
        im = C.fisheye2panoram(image, (512, 1024), offset_x=offset, x_min=0, x_max=0.25, y_min=0.3, y_max=0.9)
        cv2.imwrite(folder_out + filename, im)


    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_cubemap_earth():
    im_equirect = cv2.imread('./images/ex_fisheye/planet_eq_polytical.jpg')

    img_cube1 = P.equirect2cubemap(im_equirect,side=256,cube_format='dice')
    img_cube2 = C.equirect2cubemap(im_equirect,side=256,modif=False,dice=True)
    cv2.imwrite(folder_out + 'im_cube1.png', img_cube1)
    cv2.imwrite(folder_out + 'im_cube2.png', img_cube2)

    image_equirect2 = P.cubemap2equirect(img_cube2,512, 1024)
    cv2.imwrite(folder_out + 'im_eq2.png', image_equirect2)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_planar_earth():
    im_equirect = cv2.imread('./images/ex_fisheye/planet_eq_polytical.jpg')

    cv2.imwrite(folder_out + 'im1.png',P.equirect2planar(im_equirect, fov_deg=90, u_deg=40, v_deg=+30, out_hw=(512, 1024)))
    cv2.imwrite(folder_out + 'im2.png',P.equirect2planar(im_equirect, fov_deg=120, u_deg=0, v_deg=0, out_hw=(512, 1024)))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import tools_IO
    #tools_IO.remove_files(folder_out)

    #image = cv2.imread('./images/ex_fisheye/room04.jpg')
    #ex_room_from_fisheye(image)
    #ex_room_cubemap(image)
    #ex_room_planar_multiview(image)
    #ex_rooms_multivew('./images/ex_fisheye/Meeting2/',folder_out)

    tools_animation.folder_to_animated_gif_imageio('./images/ex_fisheye/Meeting2/', folder_out+'pano3.gif', framerate=6,resize_H=200, resize_W=200,stride=1)
    #tools_animation.folder_to_video(folder_out, folder_out+'room04.mp4', mask='*.png')
