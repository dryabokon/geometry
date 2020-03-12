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

    delta_angle = numpy.pi/64.0
    d=delta_angle

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_view((-d,0,0))
        if key == ord('W'): R.rotate_view((+d,0,0))
        if key == ord('A'): R.rotate_view((0,0,+d))
        if key == ord('D'): R.rotate_view((0,0,-d))

        if key == 328: R.translate_view((0,+10, 0))
        if key == 322: R.translate_view((0,-10, 0))
        if key == 329: R.translate_view((0,0,+10))
        if key == 323: R.translate_view((0,0,-10))

        if key == 294: R.reset_view()

        if key == 334: R.scale_model(1.04)
        if key == 333: R.scale_model(1.0/1.04)

        if key == glfw.KEY_ESCAPE:glfw.set_window_should_close(R.window,True)

        if key == ord('Z') and mods == glfw.MOD_CONTROL:
            R.my_VBO.remove_last_object()
            R.bind_VBO()

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

    if R.on_translate == True:
        delta_pos = (pos_button_start - numpy.array((xpos, ypos))) * 1.0/10
        R.translate_view((0, -delta_pos[1], -delta_pos[0]))

    if R.on_rotate == True:
        delta_angle = (pos_button_start - numpy.array((xpos, ypos))) * 1.0 * math.pi / W
        R.rotate_view((delta_angle[1],0, delta_angle[0]))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
filename_playground = './images/ex_GL/playground.obj'
filename_texture = './images/ex_GL/playground2.jpg'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_playground,filename_texture=filename_texture,W=W, H=H,do_normalize_model_file=False,projection_type='P',scale=(1,1,1))

    #R.init_mat_view_ETU(eye=(0,200,100) , target = (0,0,0), up=(0,+1, -2))
    #R.init_mat_view_ETU(eye=(0,  0,100) , target = (0,0,0), up=(0,-1,  0))

    glfw.set_key_callback(R.window, event_key)
    glfw.set_mouse_button_callback(R.window, event_button)
    glfw.set_cursor_pos_callback(R.window, event_position)
    glfw.set_window_size_callback(R.window, event_resize)

    while not glfw.window_should_close(R.window):
        R.draw()
        glfw.poll_events()
        glfw.swap_buffers(R.window)

    glfw.terminate()