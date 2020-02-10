import cv2
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import glfw
import tools_aruco
# ---------------------------------------------------------------------------------------------------------------------
image_shape = 400,600
fx,fy = 1090, 1090
principalX, principalY = fx/2, fy/2
cameraMatrix = numpy.array([[fx,0,principalX],[0,fy,principalY],[0,0,1]])
dist = numpy.zeros((1,5))
near = 1
far = 1050
marker_length = 0.1
# ---------------------------------------------------------------------------------------------------------------------
def draw(window):
    glfw.make_context_current(window)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    left = -principalX / fx
    right = (image_shape[1] - principalX) / fx
    bottom = (principalY - image_shape[0]) / fy
    top = principalY / fy
    glFrustum(left, right, bottom, top, near, far)

    rvec = numpy.array([-2.8054866, -1.071149,    0.45893607])
    tvec = numpy.array([-0.31804663, -0.56528067,  1.40848886])


    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(tools_aruco.compose_GL_MAT(rvec, tvec))
    tools_aruco.draw_native_axis(marker_length / 2)

    glPushMatrix()
    glRotatef(90, +1, 0, 0)
    tools_aruco.draw_cube(size=marker_length / 4)
    glPopMatrix()

    W,H = 640,480
    image_buffer = glReadPixels(0, 0, W, H, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(H, W, 3)
    return image
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(640, 480, "hidden window", None, None)
    image = draw(window)
    cv2.imwrite('./images/output/res.png', image)