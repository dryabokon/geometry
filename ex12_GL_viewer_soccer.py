import cv2
import sys
import math
import numpy
import tools_GL3D
import tools_render_CV
import glfw
import tools_aruco
# ----------------------------------------------------------------------------------------------------------------------
pos_button_start, pos_rotate_current = None, None
W,H = 1920,1080
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    delta_angle = numpy.pi/16.0
    d=delta_angle

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-d,0,0))
        if key == ord('W'): R.rotate_model((+d,0,0))

        if key == ord('O'): R.rotate_view((0, 0, +d))
        if key == ord('P'): R.rotate_view((0, 0, -d))

        if key == 294: R.reset_view()

        if key == 334: R.scale_model(1.04)
        if key == 333: R.scale_model(1.0/1.04)

        if key == ord('1'): R.inverce_transform_model('X')
        if key == ord('2'): R.inverce_transform_model('Y')
        if key == ord('3'): R.inverce_transform_model('Z')
        if key == ord('4'): R.inverce_screen_Y()

        if key == 327: R.transform_model('XY')
        if key == 329: R.transform_model('xy')
        if key == 324: R.transform_model('XZ')
        if key == 326: R.transform_model('xz')
        if key == 321: R.transform_model('YZ')
        if key == 323: R.transform_model('yz')
        if key == 325: R.transform_model(None)

        if key in [32,335]: R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:glfw.set_window_should_close(R.window,True)

        if key == ord('Z') and mods == glfw.MOD_CONTROL:
            R.my_VBO.remove_last_object()
            R.bind_VBO()

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_button(window, button, action, mods):
    global pos_button_start

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.PRESS   and (mods     in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.start_translation();
        pos_button_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.RELEASE and (mods     in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.stop_translation()

    if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS   and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):R.start_append()
    if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS   and (mods in [glfw.MOD_CONTROL,glfw.MOD_SHIFT]) ):R.start_remove()

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_position(window, xpos, ypos):

    if R.on_translate == True:
        delta_pos = (pos_button_start - numpy.array((xpos, ypos))) * 1.0/10
        R.translate_model((-delta_pos[0], -delta_pos[1], 0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_scroll(window, xoffset, yoffset):
    if R.projection_type=='P':
        if yoffset>0:R.translate_view(1.04)
        else        :R.translate_view(1.0/1.04)
    else:
        if yoffset>0:R.translate_ortho(1.04)
        else        :R.translate_ortho(1.0/1.04)
    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
filename_playground = './images/ex_GL/playground.obj'
#filename_playground = './images/ex_GL/box/box.obj'
filename_texture = './images/ex_GL/playground2.jpg'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_playground,filename_texture=filename_texture,W=W, H=H,do_normalize_model_file=False,projection_type='P',scale=(1,1,1))
    #R.init_perspective_view_soccer((+math.pi/16,0,0), (0,+10,-100), scale=(1, 1, 1))
    R.init_mat_view_ETU(eye=(0,0,-100),target = (0,0,0),up=(0,-1,   0))

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