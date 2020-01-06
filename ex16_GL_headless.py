#https://github.com/AntonOvsyannikov/DockerGL
# ----------------------------------------------------------------------------------------------------------------------
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# ----------------------------------------------------------------------------------------------------------------------
from OpenGL.raw.GLUT import glutSwapBuffers
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.osmesa import *
from PIL import Image
from PIL import ImageOps
# ----------------------------------------------------------------------------------------------------------------------
width, height = 300, 300
# ----------------------------------------------------------------------------------------------------------------------
def init():
    glClearColor(0.5, 0.5, 0.5, 1.0)
    glColor(0.0, 1.0, 0.0)
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glViewport(0, 0, width, height)
# ----------------------------------------------------------------------------------------------------------------------
def render():

    glClear(GL_COLOR_BUFFER_BIT)

    glBegin(GL_LINES)
    glVertex2d(-1, 0)
    glVertex2d(1, 0)
    glVertex2d(1, 0)
    glVertex2d(0.95, 0.05)
    glVertex2d(1, 0)
    glVertex2d(0.95, -0.05)
    glVertex2d(0, -1)
    glVertex2d(0, 1)
    glVertex2d(0, 1)
    glVertex2d(0.05, 0.95)
    glVertex2d(0, 1)
    glVertex2d(-0.05, 0.95)
    glEnd()

    glFlush()
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw():
    render()
    glutSwapBuffers()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if not os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
        print ('Use with: PYOPENGL_PLATFORM=osmesa')
        exit()

    ctx = OSMesaCreateContext(OSMESA_RGBA, None)

    buf = arrays.GLubyteArray.zeros((height, width, 4))
    assert(OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, width, height))
    assert(OSMesaGetCurrentContext())

    z = glGetIntegerv(GL_DEPTH_BITS)
    s = glGetIntegerv(GL_STENCIL_BITS)
    a = glGetIntegerv(GL_ACCUM_RED_BITS)

    init()
    render()

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), data)
    #image = ImageOps.flip(image)
    image.save('osmesa.png', 'PNG')
    OSMesaDestroyContext(ctx)
