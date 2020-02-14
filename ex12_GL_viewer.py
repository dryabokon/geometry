import math
import cv2
import numpy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import tools_GL3D
import glfw
# ----------------------------------------------------------------------------------------------------------------------
import tools_aruco
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
on_rotate = False
pos_rotate_start, pos_rotate_current = None, None
W,H = 800,600
# ----------------------------------------------------------------------------------------------------------------------
def key_event(window,key,scancode,action,mods):

    d=numpy.pi/16.0

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-d,0,0))
        if key == ord('W'): R.rotate_model((+d,0,0))
        if key == ord('A'): R.rotate_model((0,-d,0))
        if key == ord('D'): R.rotate_model((0,+d,0))
        if key == ord('R'): R.reset_view(reset_transform=False)

        if key == 334: R.scale_model(1.04)
        if key == 333: R.scale_model(1.0/1.04)

        if key == 294: R.reset_view()

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
def button_event(window, button, action, mods):
    global on_rotate,pos_rotate_start

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS):
        R.start_rotation()
        pos_rotate_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE):
        R.stop_rotation()

    return
# ----------------------------------------------------------------------------------------------------------------------
def position_event(window, xpos, ypos):

    if R.on_rotate == True:
        delta_angle = (pos_rotate_start-numpy.array((xpos,ypos)))*1.0*math.pi/W
        R.rotate_model((delta_angle[1], -delta_angle[0], 0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def scroll_callback(window, xoffset, yoffset):
    if yoffset>0:
        R.translate_view(1.04)
    else:
        R.translate_view(1.0/1.04)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box.obj', W=W, H=H,is_visible=True)
    #R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/face/face.obj', W=W, H=H,is_visible=True)
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/face/male_head.obj', W=W, H=H,is_visible=True)

    glfw.set_key_callback(R.window, key_event)
    glfw.set_mouse_button_callback(R.window, button_event)
    glfw.set_cursor_pos_callback(R.window, position_event)
    glfw.set_scroll_callback(R.window, scroll_callback)


    while not glfw.window_should_close(R.window):
        R.draw_GL()
        glfw.poll_events()
        glfw.swap_buffers(R.window)

    glfw.terminate()