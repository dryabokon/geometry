import cv2
import numpy
from CV import tools_fisheye
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
C = tools_fisheye.Converter()
# ----------------------------------------------------------------------------------------------------------------------
g_current_mouse_pos, g_mouse_event_code = numpy.zeros(4,dtype=int), None
g_rad_per_aper = 2.6
g_pano = 0
g_inv = False
g_angle_deg=0
g_shift_x =0
g_shift_y =0
# ----------------------------------------------------------------------------------------------------------------------
def click_handler(event, x, y, flags, param):

    global g_current_mouse_pos, g_mouse_event_code

    is_ctrl = (flags & 0x08) > 0
    is_shift = (flags & 0x10) > 0
    is_alt = (flags & 0x20) > 0

    g_current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (is_shift):
        g_mouse_event_code = 'LBUTTONDOWN_SHF'

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (is_ctrl):
        g_mouse_event_code = 'LBUTTONDOWN_CTL'

    if event == cv2.EVENT_LBUTTONUP and g_mouse_event_code == 'LBUTTONDOWN_SHF':
        g_mouse_event_code = 'LBUTTONUP_SHF'

    if event == cv2.EVENT_LBUTTONUP and g_mouse_event_code == 'LBUTTONDOWN_CTL':
        g_mouse_event_code = 'LBUTTONUP_CTL'

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (not is_shift) and (not is_ctrl):
        g_mouse_event_code = 'LBUTTONDOWN'

    elif event == cv2.EVENT_LBUTTONUP and g_mouse_event_code == 'LBUTTONDOWN':
        g_mouse_event_code = 'LBUTTONUP'

    elif event == cv2.EVENT_RBUTTONDBLCLK and g_mouse_event_code is None:
        g_mouse_event_code = 'RBUTTONDBLCLK'

    elif event == cv2.EVENT_RBUTTONDOWN and g_mouse_event_code is None:
        g_mouse_event_code = 'RBUTTONDOWN'

    elif event == cv2.EVENT_RBUTTONUP and g_mouse_event_code == 'RBUTTONDOWN':
        g_mouse_event_code = 'RBUTTONUP'

    if event == cv2.EVENT_MOUSEWHEEL:
        if is_ctrl:
            if flags > 0:
                g_mouse_event_code = 'MOUSEWHEEL_CTL_FW'
            else:
                g_mouse_event_code = 'MOUSEWHEEL_CTL_BK'
        elif is_shift:
            if flags > 0:
                g_mouse_event_code = 'MOUSEWHEEL_SHF_FW'
            else:
                g_mouse_event_code = 'MOUSEWHEEL_SHF_BK'
        elif is_alt:
            if flags > 0:
                g_mouse_event_code = 'MOUSEWHEEL_ALT_FW'
            else:
                g_mouse_event_code = 'MOUSEWHEEL_ALT_BK'
        else:
            if flags > 0:
                g_mouse_event_code = 'MOUSEWHEEL_FW'
            else:
                g_mouse_event_code = 'MOUSEWHEEL_BK'

        return
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):
    global g_pano,g_inv,g_shift_x,g_shift_y,g_angle_deg
    if key & 0xFF == 27: return -1

    if key & 0xFF == ord('p'):
        g_pano=(g_pano+1)%3
        return 1

    if key & 0xFF == ord('i'):
        g_inv=not g_inv
        return 1

    if key & 0xFF == ord('x'):
        g_shift_x=g_shift_x+1
        return 1

    if key & 0xFF == ord('X'):
        g_shift_x=g_shift_x-1
        return 1

    if key == ord('w'):
        g_angle_deg+=1
        return 1

    if key == ord('s'):
        g_angle_deg-=1
        return 1

    if key == ord('a'):
        g_angle_deg-=90
        return 1

    if key == ord('d'):
        g_angle_deg+=90
        return 1


    return 0
# ---------------------------------------------------------------------------------------------------------------------
def process_mouse():
    global g_mouse_event_code
    global g_rad_per_aper
    scale = 0.05

    if g_mouse_event_code == 'MOUSEWHEEL_FW':
        g_rad_per_aper+= scale
        g_mouse_event_code = None
        return 1

    if g_mouse_event_code == 'MOUSEWHEEL_BK':
        g_rad_per_aper-= scale
        g_mouse_event_code = None
        return 1


    return 0
# ---------------------------------------------------------------------------------------------------------------------
def draw_legend(image):

    image = tools_draw_numpy.draw_text(image, 'g_rad_per_aper = %1.1f' % g_rad_per_aper, (10, 10), (0, 0, 100))

    return image
# ---------------------------------------------------------------------------------------------------------------------
def GUI_loop():

    image0 = cv2.imread('./images/ex_fisheye/room3a.jpg')
    image0 = tools_image.do_resize(image0,(1024,1024))

    global g_shift_x,g_shift_y,g_angle_deg
    aperture = image0.shape[0]/4

    outShape = [500, 1000]
    should_be_closed = False
    should_be_refreshed = True

    while not should_be_closed:

        res = process_mouse()
        if res>0:should_be_refreshed |= True

        if should_be_refreshed:
            xx = tools_image.rotate_image(image0, g_angle_deg)
            im_equirect = C.fisheye2equirect(xx, outShape, aperture=aperture, radius=g_rad_per_aper*aperture,delx=g_shift_x,dely=g_shift_y)[:, ::-1]
            image = im_equirect
            if g_pano in [0,1]:
                image = C.equirect2cubemap(im_equirect, side=256, modif=g_pano==0, dice=True)

            if g_inv:
                image = image[::-1,::-1]

            image = draw_legend(image)
            cv2.imshow(window_name, image)
            should_be_refreshed = False

        key = cv2.waitKey(1)
        res = process_key(key)
        if res < 0: should_be_closed = True
        if res > 0: should_be_refreshed |= True

    cv2.destroyAllWindows()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    window_name = 'fisheye'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, click_handler)
    GUI_loop()

