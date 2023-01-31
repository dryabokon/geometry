# ---------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_CV3D
# ---------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------------------------------------------------
class Render_Window:

    def __init__(self,filename_obj,filename_image=None):

        self.scale = 0.75
        self.textcolor = (0, 0, 0)
        self.image_to_display0 = cv2.imread(filename_image)
        self.image_to_display = self.image_to_display0.copy()
        self.R = tools_CV3D.render_CV3D(filename_obj, self.image_to_display.shape[1], self.image_to_display.shape[0], do_normalize_model_file=False, projection_type='P',
            rvec_model=(0, 0, 0), tvec_model = (0, 0, 0), eye = (0,+0,0),target=(0,1,0),up=(0,0,1))

        self.refresh_image()

        return
# ---------------------------------------------------------------------------------------------------------------------
    def draw_legend(self, image):
        cv2.rectangle(image, (0, 0), (300, 48), (128, 128, 128), -1)
        return image

# ---------------------------------------------------------------------------------------------------------------------
    def refresh_image(self):
        self.image_to_display = self.image_to_display0.copy()

        self.image_to_display = self.R.draw_debug_info(self.image_to_display)

        self.image_to_display = tools_image.desaturate(self.image_to_display, level=0.25)
        self.image_to_display = self.draw_legend(self.image_to_display)


        return
# ---------------------------------------------------------------------------------------------------------------------
# =====================================================================================================================
g_coord, g_current_mouse_pos, g_mouse_event_code = [], numpy.zeros(4,dtype=numpy.int), None
RW = Render_Window(filename_image='./images/ex_GL/nuscene/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151604037558.jpg',filename_obj='./images/output/lidar.obj')
# ---------------------------------------------------------------------------------------------------------------------
def click_handler(event, x, y, flags, param):

    global g_coord, g_current_mouse_pos, g_mouse_event_code

    return
# ---------------------------------------------------------------------------------------------------------------------
def process_mouse():
    global g_coord, g_current_mouse_pos, g_mouse_event_code

    return 0
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    delta_angle = numpy.pi / 16.0
    t=5

    if key & 0xFF == 27: return -1

    if key == ord('w'):
        RW.R.translate_view((0,0,+t))
        return 1
    if key == ord('s'):
        RW.R.translate_view((0,0,-t))
        return 1

    if key == ord('d'):
        RW.R.translate_view((-t,0,0))
        return 1

    if key == ord('a'):
        RW.R.translate_view((+t,0,0))
        return 1
    if key == ord('q'):
        RW.R.translate_view((0,-t,0))
        return 1
    if key == ord('e'):
        RW.R.translate_view((0,+t,0))
        return 1

    if key == 326:
        RW.R.rotate_view((0,0,-delta_angle))
        return 1

    if key == 324:
        RW.R.rotate_view((0,0,+delta_angle))
        return 1
    if key == 328:
        RW.R.rotate_view((+delta_angle,0,0))
        return 1
    if key == 322:
        RW.R.rotate_view((-delta_angle,0,0))
        return 1

    if key == 341:
        RW.R.ctrl_pressed = True

    if key == 294:
        RW.R.reset_view()
        return 1

    if key == 334: RW.R.scale_model_vector((1.04, 1.04, 1.04))
    if key == 333: RW.R.scale_model_vector((1.0 / 1.04, 1.0 / 1.04, 1.0 / 1.04))

    #if key in [32, 335]: RW.R.stage_data(folder_out)

    return 0
# ---------------------------------------------------------------------------------------------------------------------
def application_loop():

    should_be_closed = False
    should_be_refreshed = True

    while not should_be_closed:

        res = process_mouse()
        if res>0:
            should_be_refreshed |= True

        key = cv2.waitKey(1)
        res = process_key(key)
        if res < 0: should_be_closed = True
        if res > 0: should_be_refreshed |= True

        if should_be_refreshed:
            RW.refresh_image()
            resized = cv2.resize(RW.image_to_display,(int(RW.scale*RW.image_to_display.shape[1]),int(RW.scale*RW.image_to_display.shape[0])))
            cv2.imshow(window_name, resized)
            should_be_refreshed = False

    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    window_name = 'CV_markup'
    cv2.namedWindow(window_name,cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, click_handler)
    cv2.resizeWindow(window_name, int(RW.scale*RW.image_to_display.shape[1]),int(RW.scale*RW.image_to_display.shape[0]))
    application_loop()

