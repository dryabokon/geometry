import cv2
import numpy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
# ----------------------------------------------------------------------------------------------------------------------
import tools_aruco
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
image_shape = 400,600
fx,fy = 1090, 1090
principalX, principalY = fx/2, fy/2
cameraMatrix = numpy.array([[fx,0,principalX],[0,fy,principalY],[0,0,1]])
dist = numpy.zeros((1,5))
near = 1
far = 1050
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
frame = []
# ----------------------------------------------------------------------------------------------------------------------
lightZeroPosition = [10.0, 10.0, 10.0, 1.0]
lightZeroColor = [2.5, 2.5, 2.5, 1]
# ----------------------------------------------------------------------------------------------------------------------
filename_in = 'images/ex_aruco/01.jpg'
filename_out = 'images/output/ar.png'
USE_CAMERA = False
# ----------------------------------------------------------------------------------------------------------------------
if USE_CAMERA:
    cap = cv2.VideoCapture(0)
# ----------------------------------------------------------------------------------------------------------------------
def keyboard(key, x, y):

    if (key.decode("utf-8") == 'q') or (key == b'\x1b'):
        if USE_CAMERA:
            cap.release()
        cv2.destroyAllWindows()
        cv2.imwrite(filename_out, frame)
        exit()

    return
# ----------------------------------------------------------------------------------------------------------------------
def draw():

    global frame

    if USE_CAMERA:
        ret, frame = cap.read()
    else:
        frame = cv2.imread(filename_in)

    frame = tools_image.desaturate(frame)

    frame, rvec, tvec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, cameraMatrix, dist)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glDrawPixels(frame.shape[1], frame.shape[0], GL_BGR, GL_UNSIGNED_BYTE, cv2.flip(frame, 0).data)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    left = -principalX / fx
    right = (image_shape[1] - principalX) / fx
    bottom = (principalY - image_shape[0]) / fy
    top = principalY / fy
    glFrustum(left, right, bottom, top,near, far)

    #a = (GLfloat * 16)()
    #glGetFloatv(GL_PROJECTION_MATRIX, a)
    #a= list(a)

    if numpy.count_nonzero(rvec)>0:
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(tools_aruco.compose_GL_MAT(rvec, tvec))
        tools_aruco.draw_native_axis(marker_length/2)

        glPushMatrix()
        glRotatef(90, +1, 0, 0)
        glutSolidTeapot(marker_length/4)#tools_aruco.draw_cube(size=marker_length/4)
        glPopMatrix()


    glutSwapBuffers()
    glFlush()
    glutPostRedisplay()
    return True
# ----------------------------------------------------------------------------------------------------------------------
def reshape(w, h):
    glViewport(0, 0, w, h)
# ----------------------------------------------------------------------------------------------------------------------
def idle():
    glutPostRedisplay()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(image_shape[1], image_shape[0])

    glutCreateWindow("OpenGL")

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0])

    glutDisplayFunc(draw)
    glutKeyboardFunc(keyboard)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutMainLoop()
