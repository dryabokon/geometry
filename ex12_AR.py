import pyrr
import cv2
import numpy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
# ----------------------------------------------------------------------------------------------------------------------
import tools_render_CV
import tools_aruco
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
marker_length = 0.1
# ----------------------------------------------------------------------------------------------------------------------
filename_in = 'images/ex_aruco/01.jpg'
filename_out = 'images/output/ar.png'
USE_CAMERA = False
# ----------------------------------------------------------------------------------------------------------------------
if USE_CAMERA:
    cap = cv2.VideoCapture(0)
# ----------------------------------------------------------------------------------------------------------------------
class render_GL(object):

    def __init__(self, W=640, H=480, scale=1):

        self.W = W
        self.H = H
        self.rvec, self.tvec = (0,0,0),(0,0,0)
        self.bg_color = (0.3, 0.3, 0.3, 0.5)

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(W, H)

        glutCreateWindow("OpenGL")

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [2.5, 2.5, 2.5, 1])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0])
        glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], self.bg_color[3])

        glutDisplayFunc(self.draw)
        glutKeyboardFunc(self.keyboard)
        glutReshapeFunc(self.reshape)
        glutIdleFunc(self.idle)
        glutMainLoop()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def keyboard(self,key, x, y):

        if (key.decode("utf-8") == 'q') or (key == b'\x1b'):
            if USE_CAMERA:
                cap.release()
            cv2.destroyAllWindows()
            cv2.imwrite(filename_out, frame)
            exit()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_mat_projection0(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fx, fy = float(2 * self.W), float(2 * self.H)
        left,right,bottom,top = -0.5, (self.W - fx/2) / fx, (fy/2 - self.H) / fy, 0.5
        near, far = 1, 1000

        glFrustum(left, right, bottom, top, near, far)

        a = (GLfloat * 16)()
        glGetFloatv(GL_PROJECTION_MATRIX, a)

        self.mat_projection = numpy.array(list(a)).reshape(4,4)
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_mat_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fx, fy = float(1 * self.W), float(1 * self.H)
        left, right, bottom, top = -0.5, (self.W - fx / 2) / fx, (fy / 2 - self.H) / fy, 0.5
        near, far = 1, 1000

        glFrustum(left, right, bottom, top, near, far)

        a = (GLfloat * 16)()
        glGetFloatv(GL_PROJECTION_MATRIX, a)

        self.mat_projection = numpy.array(list(a)).reshape(4, 4)
        self.mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
        return

# ----------------------------------------------------------------------------------------------------------------------
    def __init_mat_modelvew(self):
        R = pyrr.matrix44.create_from_eulers(self.rvec)
        T = pyrr.matrix44.create_from_translation(self.tvec)
        self.mat_view = pyrr.matrix44.multiply(R, T)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw(self):

        global frame

        if USE_CAMERA:
            ret, frame = cap.read()
        else:
            frame = cv2.imread(filename_in)


        self.init_mat_projection0()

        frame = tools_image.desaturate(frame)
        frame, self.rvec, self.tvec = tools_aruco.detect_marker_and_draw_axes(frame, marker_length, self.mat_camera, numpy.zeros(4))

        self.__init_mat_modelvew()

        self.draw_mat(self.mat_projection, 20, 20 , frame)
        self.draw_mat(self.mat_view      , 20, 120, frame)
        self.draw_mat(self.mat_camera    , 20, 350, frame)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDrawPixels(frame.shape[1], frame.shape[0], GL_BGR, GL_UNSIGNED_BYTE, cv2.flip(frame, 0).data)
        glEnable(GL_DEPTH_TEST)

        if numpy.count_nonzero(self.rvec)>0:
            glMatrixMode(GL_MODELVIEW)
            glLoadMatrixf(self.mat_view)
            tools_aruco.draw_native_axis(marker_length/2)
            glPushMatrix()
            tools_aruco.draw_cube(size=marker_length/4)
            glPopMatrix()

        glutSwapBuffers()
        glFlush()
        glutPostRedisplay()
        return True
# ----------------------------------------------------------------------------------------------------------------------
    def reshape(self,w, h):
        glViewport(0, 0, w, h)
# ----------------------------------------------------------------------------------------------------------------------
    def idle(self):
        glutPostRedisplay()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_text(self, message, pos_x, pos_y):
        glColor4f(100.0, 250, 0.1, 0)
        glRasterPos2f(pos_x, pos_y)
        for key in message:
            glutBitmapCharacter(OpenGL.GLUT.fonts.GLUT_BITMAP_HELVETICA_10, ctypes.c_int(ord(key)))
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def draw_mat(self, M, posx, posy, image):
        for row in range(M.shape[0]):
            if M.shape[1]==4:
                string1 = '%+1.2f %+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2], M[row, 3])
            else:
                string1 = '%+1.2f %+1.2f %+1.2f' % (M[row, 0], M[row, 1], M[row, 2])
            image = cv2.putText(image, '{0}'.format(string1), (posx, posy + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1, cv2.LINE_AA)
        return image
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    R = render_GL()
