import math
import numpy
import tools_GL3D
import glfw
# ----------------------------------------------------------------------------------------------------------------------
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
pos_button_start, pos_rotate_current = None, None
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    delta_angle = numpy.pi/32.0
    t=0.5

    if key == ord('W'): R.translate_view((0, 0, +t))
    if key == ord('S'): R.translate_view((0, 0, -t))
    if key == ord('D'): R.translate_view((-t, 0, 0))
    if key == ord('A'): R.translate_view((+t, 0, 0))
    if key == ord('Q'): R.translate_view((0, -t, 0))
    if key == ord('E'): R.translate_view((0, +t, 0))
    if key == 326: R.rotate_view((0,0,-delta_angle))
    if key == 324: R.rotate_view((0,0,+delta_angle))
    if key == 328: R.rotate_view((+delta_angle,0,0))
    if key == 322: R.rotate_view((-delta_angle,0,0))

    if action == glfw.PRESS:

        if key==341:R.ctrl_pressed = True

        if key == 294:
            R.reset_view()

        if key == ord('L'):
            R.update_background('./images/ex_GL/nuscene/board.obj')
            R.wired_mode=(R.wired_mode+1)%2
            R.bind_VBO(R.wired_mode)

        if key in [32,335]:
            R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:glfw.set_window_should_close(R.window,True)

    if (action ==glfw.RELEASE) and (key == 341):R.ctrl_pressed = False

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_button(window, button, action, mods):
    global pos_button_start

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.PRESS   and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):R.start_rotation();pos_button_start = glfw.get_cursor_pos(window)
    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.RELEASE and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):R.stop_rotation()

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.PRESS   and (mods     in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.start_translation()
        pos_button_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.RELEASE and (mods     in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.stop_translation()


    return
# ----------------------------------------------------------------------------------------------------------------------
def event_position(window, xpos, ypos):

    if R.on_rotate == True:
        delta_angle = (pos_button_start - numpy.array((xpos, ypos))) * 1.0 * math.pi / W
        R.rotate_view((-delta_angle[1],0,-delta_angle[0]))

    if R.on_translate == True:
        delta_pos = (pos_button_start - numpy.array((xpos, ypos))) * 1.0/2
        R.translate_view((-delta_pos[0], +delta_pos[1], 0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_scroll(window, xoffset, yoffset):

    if yoffset>0:R.scale_projection(1.10)
    else        :R.scale_projection(1.0 / 1.10)

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
# filename_obj     = './images/ex_GL/box/box_2.obj'
# rvec_obj = (0.0, 0.0, 0.0)
# tvec_obj = (0, 10, 0)
# M_obj = tools_pr_geom.compose_RT_mat(rvec_obj, tvec_obj, do_rodriges=False,do_flip=False, GL_style=True)
# textured = True
# do_normalize_model_file = False
# eye = (0,0,0)
# target=(0,1,0)
# up=(0,0,1)
# ----------------------------------------------------------------------------------------------------------------------
filename_obj     = './images/ex_GL/nuscene/lidar3.obj'
do_normalize_model_file = False
rvec_obj = (0.0, 0.0, 0)
tvec_obj=(0,0,0)
M_obj = tools_pr_geom.compose_RT_mat(rvec_obj, tvec_obj, do_rodriges=False,do_flip=False, GL_style=True)
textured = True
eye = (0,0,0)
target=(0,1,0)
up=(0,0,1)
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/gl/'
W,H = 1600,900
cam_fov_deg = 90
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=W, H=H, do_normalize_model_file=do_normalize_model_file,textured=textured,projection_type='P',
                               cam_fov_deg=cam_fov_deg,scale=(1, 1, 1),eye = eye,target=target,up=up,
                               M_obj=M_obj)

    glfw.set_key_callback(R.window, event_key)
    glfw.set_mouse_button_callback(R.window, event_button)
    glfw.set_cursor_pos_callback(R.window, event_position)
    glfw.set_scroll_callback(R.window, event_scroll)
    glfw.set_window_size_callback(R.window, event_resize)

    while not glfw.window_should_close(R.window):
        R.draw()
        glfw.poll_events()
        glfw.swap_buffers(R.window)

    glfw.terminate()
    