import sys
import math
import numpy
import tools_GL3D
import tools_render_CV
import glfw
# ----------------------------------------------------------------------------------------------------------------------
from tools_wavefront import ObjLoader
# ----------------------------------------------------------------------------------------------------------------------
pos_rotate_start, pos_rotate_current = None, None
#W,H = 800,800
W,H = 912,1024
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):


    delta_angle = numpy.pi/16.0
    d=delta_angle

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-d,0,0))
        if key == ord('W'): R.rotate_model((+d,0,0))
        if key == ord('A'): R.rotate_model((0,-d,0))
        if key == ord('D'): R.rotate_model((0,+d,0))
        #if key == ord('Z'): R.rotate_model((0,0,-d))
        #if key == ord('X'): R.rotate_model((0,0,+d))

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

        if key == 291: R.save_markers(folder_out+'markers.txt')
        if key == 292: R.load_markers(folder_out + 'markers.txt',filename_marker_obj)

        if key in [32,335]: R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(R.window,True)

        if key == ord('Z') and mods == glfw.MOD_CONTROL:
            R.my_VBO.remove_last_object()
            R.bind_VBO()

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_button(window, button, action, mods):
    global pos_rotate_start

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS):
        R.start_rotation()
        pos_rotate_start = glfw.get_cursor_pos(window)

    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE):
        R.stop_rotation()

    if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):
        R.start_append()

    if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS and (mods in [glfw.MOD_CONTROL,glfw.MOD_SHIFT]) ):
        R.start_remove()

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_position(window, xpos, ypos):

    if R.on_rotate == True:
        delta_angle = (pos_rotate_start-numpy.array((xpos,ypos)))*1.0*math.pi/W
        R.rotate_model((delta_angle[1], -delta_angle[0], 0))

    if R.on_append == True:
        R.stop_append()
        point_2d = (W - xpos, ypos)
        ray_begin, ray_end = tools_render_CV.get_ray(point_2d, numpy.full((H, W, 3), 76, dtype=numpy.uint8),R.mat_projection, R.mat_view, R.mat_model, R.mat_trns)
        tvec = tools_render_CV.get_interception_ray_triangles(ray_begin, ray_end - ray_begin, R.object.coord_vert,R.object.coord_norm,R.object.idx_vertex,R.object.idx_normal)
        if tvec is not None:
            R.my_VBO.append_object(filename_marker_obj, (0.7, 0.2, 0), do_normalize_model_file=True, svec=(R.marker_scale, R.marker_scale, R.marker_scale), tvec=tvec)
            R.bind_VBO()

    if R.on_remove == True:
        R.stop_remove()
        R.my_VBO.remove_last_object()

        R.bind_VBO()

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
filename_box     = './images/ex_GL/box/box.obj'
filename_marker_obj    = './images/ex_GL/sphere/sphere.obj'
# ----------------------------------------------------------------------------------------------------------------------
filename_head_obj1 = './images/ex_GL/face/face_norm.obj'
filename_markers1 ='./images/ex_GL/face/markers_face_norm.txt'
# ----------------------------------------------------------------------------------------------------------------------
filename_head_obj2 = './images/ex_GL/face/male_head_exp_norm.obj'
filename_markers2 ='./images/ex_GL/face/markers_male_head_exp_norm.txt'
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    R = tools_GL3D.render_GL3D(filename_obj=filename_head_obj2, W=W, H=H)

    #R.load_markers(filename_markers, filename_marker_obj,0.015)
    #rvec, tvec = numpy.array([-0.08, -math.pi, -0.02]),numpy.array([-0.03, -0.19, 2.4])
    #R.init_mat_view_RT(rvec, tvec, flip=True)

    R.my_VBO.append_object(filename_head_obj1, (0.7, 0.2, 0), do_normalize_model_file=False, svec=(1, 1, 1), tvec=(0, 0, 0))
    R.bind_VBO()

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