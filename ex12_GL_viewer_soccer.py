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
W,H = 1280,720
folder_out = './images/output/gl/'
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    delta_angle = numpy.pi/64.0
    d=delta_angle

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_view((-d,0,0))
        if key == ord('W'): R.rotate_view((+d,0,0))
        if key == ord('A'): R.rotate_view((0,0,+d))
        if key == ord('D'): R.rotate_view((0,0,-d))
        if key == ord('Z'): R.rotate_view((0,+d,0))
        if key == ord('X'): R.rotate_view((0,-d,0))

        if key == 328: R.translate_view((0,+10, 0))
        if key == 322: R.translate_view((0,-10, 0))
        if key == 329: R.translate_view((0,0,+10))
        if key == 323: R.translate_view((0,0,-10))
        if key == 324: R.translate_view((+10,0,0))
        if key == 326: R.translate_view((-10,0,0))
        if key == 325:
            #for c in range(4):
            R.center_view()


        if key == 294: R.reset_view()

        if key == 334: R.scale_model_vector((1.05, 1.05, 1.05))
        if key == 333: R.scale_model_vector((1.0/1.05, 1.0/1.05, 1.0/1.05))

        if key in [32, 335]: R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:glfw.set_window_should_close(R.window,True)


    return
# ----------------------------------------------------------------------------------------------------------------------
def event_button(window, button, action, mods):
    global pos_button_start

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.PRESS   and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.start_rotation()
        pos_button_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT  and action == glfw.RELEASE and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.stop_rotation()


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
        R.translate_view((0, delta_pos[1], -delta_pos[0]*0))

    if R.on_rotate == True:
        delta_angle = (pos_button_start - numpy.array((xpos, ypos))) * 1 * math.pi / W
        R.rotate_view((-delta_angle[1], 0,-delta_angle[0]))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_scroll(window, xoffset, yoffset):
    if R.projection_type=='P':
        if yoffset>0:R.scale_projection(1.12)
        else        :R.scale_projection(1.0 / 1.12)
    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
filename_obj = './images/ex_GL/pg.obj'
#filename_obj     = './images/ex_GL/box/box2_simple.obj'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj,W=W, H=H,do_normalize_model_file=False,projection_type='P',scale=(1,1,1))

    R.init_mat_view_ETU(eye=(0,0,20) , target = (0,0,0), up=(0,-1, -2))
    R.center_view()
    #R.init_perspective_view([-0.49, -0.45, -1.02], [-45.85, 3.37, 2.30], 1.54,1.54)

    #RT = numpy.array([[    1.  ,    -0.01,  -0.01,     0.  ],[    0.01 ,   -0.33,   0.95 ,    0.  ],[   -0.02 ,   -0.95,  -0.33 ,    0.  ],[ -560.65 ,  123.73,  -1043.38 ,    1.  ]])
    #image = R.get_image_perspective_M(RT, lookback=False,do_debug=True)
    #cv2.imwrite(folder_out + 'GT_M.png', image)

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