import sys
import math
import numpy
import tools_GL3D
import glfw
from tools_wavefront import ObjLoader
# ----------------------------------------------------------------------------------------------------------------------
pos_rotate_start, pos_rotate_current = None, None
W,H = 800,800
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    d=numpy.pi/16.0

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-d,0,0))
        if key == ord('W'): R.rotate_model((+d,0,0))
        if key == ord('A'): R.rotate_model((0,-d,0))
        if key == ord('D'): R.rotate_model((0,+d,0))
        if key == ord('R'): R.reset_view()

        if key == 334: R.scale_model(1.04)
        if key == 333: R.scale_model(1.0/1.04)

        if key == 294:R.reset_view()

        if key == 327: R.transform_model('XY')
        if key == 329: R.transform_model('xy')
        if key == 324: R.transform_model('XZ')
        if key == 326: R.transform_model('xz')
        if key == 321: R.transform_model('YZ')
        if key == 323: R.transform_model('yz')
        if key == 325: R.transform_model(None)


    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(R.window,True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_button(window, button, action, mods):
    global pos_rotate_start

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS):
        R.start_rotation()
        pos_rotate_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE):
        R.stop_rotation()

    #if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS):
    #    R.append_object(filename_sphere,(0.5,0,0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_position(window, xpos, ypos):

    if R.on_rotate == True:
        delta_angle = (pos_rotate_start-numpy.array((xpos,ypos)))*1.0*math.pi/W
        R.rotate_model((delta_angle[1], -delta_angle[0], 0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_scroll(window, xoffset, yoffset):
    if yoffset>0:
        R.translate_view(1.04)
    else:
        R.translate_view(1.0/1.04)
    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_convert(filename_in,filename_out):
    Obj = ObjLoader()
    Obj.convert(filename_in,filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
filename_out = './images/ex_GL/face/male_head_exp.obj'
# ----------------------------------------------------------------------------------------------------------------------
#filename_in = './images/ex_GL/rock/TheRock2.obj'
filename_out = './images/ex_GL/rock/TheRock2_exp.obj'
# ----------------------------------------------------------------------------------------------------------------------
#filename_in = './images/ex_GL/sphere/sphere.obj'
#filename_out = './images/ex_GL/sphere/sphere_exp.obj'
# ----------------------------------------------------------------------------------------------------------------------
filename_sphere = './images/ex_GL/sphere/sphere.obj'
filename_box = './images/ex_GL/box/box.obj'
filename_face = './images/ex_GL/face/face.obj'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_convert(filename_in,filename_out)
    #sys.exit(1)

    R = tools_GL3D.render_GL3D(filename_obj=filename_face, W=W, H=H)

    glfw.set_key_callback(R.window, event_key)
    glfw.set_mouse_button_callback(R.window, event_button)
    glfw.set_cursor_pos_callback(R.window, event_position)
    glfw.set_scroll_callback(R.window, event_scroll)
    glfw.set_window_size_callback(R.window, event_resize)

    R.transform_model('xz')

    while not glfw.window_should_close(R.window):
        R.draw()
        glfw.poll_events()
        glfw.swap_buffers(R.window)

    glfw.terminate()