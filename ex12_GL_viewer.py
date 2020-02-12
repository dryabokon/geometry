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
def key_event(window,key,scancode,action,mods):
    if action == glfw.PRESS:
        if key == 326:R.rotate_light((0,+0.1,0))
        if key == 324:R.rotate_light((0,-0.1,0))
        if key == 328:R.rotate_light((+0.1,0,0))
        if key == 322:R.rotate_light((-0.1,0,0))

    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(R.window,True)


    return
# ----------------------------------------------------------------------------------------------------------------------
def mouse_event(window,button, action, mods):
    #if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS):


    return
# ----------------------------------------------------------------------------------------------------------------------
def scroll_callback(window, xoffset, yoffset):
    if yoffset>0:
        R.translate_Z_view(1.04)
    else:
        R.translate_Z_view(1.0/1.04)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/box/box.obj', W=800, H=600,is_visible=True)
    R = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/face/male_head.obj', W=800, H=600,is_visible=True)

    glfw.set_key_callback(R.window, key_event)
    glfw.set_mouse_button_callback(R.window, mouse_event)
    glfw.set_scroll_callback(R.window, scroll_callback)

    while not glfw.window_should_close(R.window):
        R.draw_GL()
        glfw.poll_events()
        glfw.swap_buffers(R.window)

    glfw.terminate()