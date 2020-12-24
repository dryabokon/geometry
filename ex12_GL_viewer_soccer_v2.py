import math
import numpy
import tools_GL3D
import glfw
# ----------------------------------------------------------------------------------------------------------------------
pos_button_start, pos_rotate_current = None, None
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    delta_angle = numpy.pi/16.0
    d=delta_angle


    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-d,0,0))
        if key == ord('W'): R.rotate_model((+d,0,0))
        if key == ord('A'): R.rotate_model((0,+d,0))
        if key == ord('D'): R.rotate_model((0,-d,0))
        if key == ord('Z'): R.rotate_model((0,0,-d))
        if key == ord('X'): R.rotate_model((0,0,+d))

        if key == 328: R.translate_model((0,+10, 0))
        if key == 322: R.translate_model((0,-10, 0))
        if key == 329: R.translate_model((0,0,+10))
        if key == 323: R.translate_model((0,0,-10))
        #if key == 324: R.translate_model((+10,0,0))
        #if key == 326: R.translate_model((-10,0,0))


        if key==341:
            R.ctrl_pressed = True

        if key == 294:
            R.reset_view()
            R.init_mat_view_ETU((0, 0, 0), (0, 0, -1), (0, -1, 0))
            R.rotate_model((2*numpy.pi/3, 0, 0))

        if key in [32,335]: R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:glfw.set_window_should_close(R.window,True)

    if action ==glfw.RELEASE:
        if key == 341:
            R.ctrl_pressed = False

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
        R.rotate_model((+delta_angle[1], +delta_angle[0], 0))

    if R.on_translate == True:
        delta_pos = (pos_button_start - numpy.array((xpos, ypos))) * 1.0/2
        R.translate_model((0, -delta_pos[1], delta_pos[0]))

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
filename_box     = './images/ex_GL/soccer/soccer_field2.obj'
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/gl/'
W,H = 1280,720
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_box, W=W, H=H, do_normalize_model_file=False, projection_type='P',scale=(1, 1, 1),tvec=(0,50,50))
    R.init_mat_view_ETU((0,0,0), (0,0,-1), (0,-1,0))
    R.rotate_model((2*numpy.pi/3,0,0))

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