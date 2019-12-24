#https://www.andersriggelsen.dk/glblendfunc.php
#http://jerome.jouvie.free.fr/opengl-tutorials/Tutorial9.php
# ----------------------------------------------------------------------------------------------------------------------
import cv2
import time
import numpy
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from PIL import Image
import tools_image
import detector_landmarks
import glfw
from scipy.spatial import Delaunay
import tools_calibrate
# ----------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ----------------------------------------------------------------------------------------------------------------------
use_camera = True
do_transfer = True
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_faceswap/01/'
list_filenames   = ['person1.jpg','person2.jpg','person3.jpg','person4.jpg','person5.jpg','person6.jpg']
filename_out = 'images/output/ar.png'
USE_CAMERA = False
# ----------------------------------------------------------------------------------------------------------------------
window = 0
# ----------------------------------------------------------------------------------------------------------------------
def refresh2d(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    return
# ----------------------------------------------------------------------------------------------------------------------
def initTexture(inputimage):

    image = Image.fromarray(inputimage)
    width = image.size[0]
    height = image.size[1]

    image = image.tobytes("raw", "RGBX", 0, -1)
    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)  # 2d texture (x and y size)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)


    texture =  gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)

    return texture
# ----------------------------------------------------------------------------------------------------------------------
def draw_image(window,window_H, window_W,image_texture):

    initTexture(image_texture[:, :, [2, 1, 0]])

    texture_height, texture_width, _ = image_texture.shape

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0);glVertex2f(0, 0)
    glTexCoord2f(1, 0);glVertex2f(texture_width, 0)
    glTexCoord2f(1, 1);glVertex2f(texture_width, texture_height)
    glTexCoord2f(0, 1);glVertex2f(0, texture_height)

    glEnd()
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_morphed_mesh(window, window_H, window_W, points_coord,points_text_coord, triangles,image_texture):


    initTexture(image_texture[:, :, [2, 1, 0]])
    glBegin(GL_TRIANGLES)

    texture_height, texture_width, _ = image_texture.shape
    for triangle in triangles:
        x0 = points_coord[triangle[0], 0]
        y0 = points_coord[triangle[0], 1]
        x1 = points_coord[triangle[1], 0]
        y1 = points_coord[triangle[1], 1]
        x2 = points_coord[triangle[2], 0]
        y2 = points_coord[triangle[2], 1]

        tx0 = points_text_coord[triangle[0], 0]
        ty0 = points_text_coord[triangle[0], 1]
        tx1 = points_text_coord[triangle[1], 0]
        ty1 = points_text_coord[triangle[1], 1]
        tx2 = points_text_coord[triangle[2], 0]
        ty2 = points_text_coord[triangle[2], 1]

        glTexCoord2f(float(tx0 / texture_width), texture_height - float(ty0 / texture_height))
        glVertex2f(x0, window_H-y0)

        glTexCoord2f(float(tx1 / texture_width), texture_height - float(ty1 / texture_height))
        glVertex2f(x1, window_H-y1)

        glTexCoord2f(float(tx2 / texture_width), texture_height - float(ty2 / texture_height))
        glVertex2f(x2, window_H-y2)

    glEnd()
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_transfered_image(window, window_H, window_W, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C):

    H = tools_calibrate.get_transform_by_keypoints(L_clbrt, L_actor)
    if H is None:
        return image_actor

    glfw.make_context_current(window)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    refresh2d(window_W, window_H)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)

    LC_aligned, LA_aligned = tools_calibrate.translate_coordinates(image_clbrt, image_actor, H, L_clbrt, L_actor)

    #glBlendFunc(GL_ONE, GL_ZERO)# disable blending
    draw_image(window, window_H, window_W,image_actor)

    draw_morphed_mesh(window, window_H, window_W, LA_aligned, L_clbrt, del_triangles_C, image_clbrt)

    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)   #defines blending factors
    #glColor4f(0.4, 1.0, 1.0, 0.50)                      #defines the alpha value = 0.75f

    L2_aligned_mouth = LA_aligned[numpy.arange(48, 61, 1).tolist()]
    del_mouth = Delaunay(L2_aligned_mouth).vertices
    draw_morphed_mesh(window, window_H, window_W, L2_aligned_mouth, L2_aligned_mouth,del_mouth,image_actor)

    return
# ----------------------------------------------------------------------------------------------------------------------
def key_callback(window, key, scancode, action, mods):

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window,1)

    return
# ----------------------------------------------------------------------------------------------------------------------
def main():
    window_W, window_H = 640, 480

    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(3, window_H)
        cap.set(4, window_W)

    glfw.init()
    window = glfw.create_window(window_W, window_H, "Face Swap", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    filename_clbrt = list_filenames[0]
    filename_actor = list_filenames[1]

    image_clbrt = cv2.imread(folder_in + filename_clbrt)
    image_actor = cv2.imread(folder_in + filename_actor)


    L_clbrt = D.get_landmarks(image_clbrt)
    del_triangles_C = Delaunay(L_clbrt).vertices
    L_actor = D.get_landmarks(image_actor)

    cnt, start_time, fps = 0, time.time(), 0
    while not glfw.window_should_close(window):
        if use_camera:
            ret, image_actor = cap.read()
            L_actor = D.get_landmarks(image_actor)


        window_W, window_H = image_actor.shape[1], image_actor.shape[0]
        glfw.set_window_size(window, window_W, window_H)

        if do_transfer:
            draw_transfered_image(window, window_H, window_W, image_clbrt, image_actor, L_clbrt, L_actor, del_triangles_C)
        else:
            draw_image(window, window_H, window_W,image_actor)

        glfw.swap_buffers(window)
        glfw.poll_events()
        cnt += 1
        if time.time() > start_time: fps = cnt / (time.time() - start_time)

    glfw.terminate()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()