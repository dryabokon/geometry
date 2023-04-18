import math
import numpy
import glfw
# ----------------------------------------------------------------------------------------------------------------------
import tools_GL3D
from CV import tools_pr_geom
import tools_wavefront
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
pos_button_start, pos_rotate_current = None, None
# ----------------------------------------------------------------------------------------------------------------------
def event_key(window, key, scancode, action, mods):

    delta_angle = numpy.pi/16.0
    t=5.0

    if action == glfw.PRESS:

        if key == ord('S'): R.rotate_model((-delta_angle,0,0))
        if key == ord('W'): R.rotate_model((+delta_angle,0,0))
        if key == ord('A'): R.rotate_model((0,+delta_angle,0))
        if key == ord('D'): R.rotate_model((0,-delta_angle,0))
        if key == ord('Z'):
            if not R.ctrl_pressed:
                R.rotate_model((0,0,-delta_angle))
        if key == ord('X'): R.rotate_model((0,0,+delta_angle))

        if key == ord('O'): R.rotate_view((0,0,+delta_angle))
        if key == ord('P'): R.rotate_view((0,0,-delta_angle))
        if key == ord('I'): R.rotate_view((+delta_angle,0,0))
        if key == ord('K'): R.rotate_view((-delta_angle,0,0))


        #numpad
        if key == 327: R.transform_model('XY')
        if key == 329: R.transform_model('xy')
        if key == 324: R.transform_model('XZ')
        if key == 326: R.transform_model('xz')
        if key == 321: R.transform_model('YZ')
        if key == 323: R.transform_model('yz')

        if key==341:
            R.ctrl_pressed = True

        if key == 294: R.reset_view()

        if key == 334: R.scale_model_vector((1.04,1.04,1.04))
        if key == 333: R.scale_model_vector((1.0/1.04,1.0/1.04,1.0/1.04))

        if key == ord('1'): R.inverce_transform_model('X')
        if key == ord('2'): R.inverce_transform_model('Y')
        if key == ord('3'): R.inverce_transform_model('Z')

        if key == ord('L'):
            R.wired_mode=(R.wired_mode+1)%3
            R.bind_VBO(R.wired_mode)

        if (key == ord('Z') and R.ctrl_pressed) or (key== glfw.KEY_BACKSPACE):
            R.my_VBO.remove_last_object()

        if key in [32,335]: R.stage_data(folder_out)

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(R.window,True)

    if action ==glfw.RELEASE:
        if key == 341:
            R.ctrl_pressed = False

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

    if (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS   and (mods not in [glfw.MOD_CONTROL,glfw.MOD_SHIFT])):

        point_2d = (glfw.get_cursor_pos(window)[0], glfw.get_cursor_pos(window)[1])
        ray_begin, ray_end = tools_render_CV.get_ray(point_2d, numpy.full((R.H, R.W, 3), 32, dtype=numpy.uint8), R.mat_projection, R.mat_view, R.mat_model,R.mat_trns)
        collisions_3d = tools_render_CV.get_interceptions_ray_triangles(ray_begin, (ray_begin - ray_end), R.object.coord_vert, R.object.idx_vertex)
        collision_3d = R.get_best_collision(collisions_3d)

        if collision_3d is not None:
            M = tools_pr_geom.compose_RT_mat((0,0,0), collision_3d, do_rodriges=False, do_flip=False,GL_style=True)
            R.my_VBO.append_object('./images/ex_GL/sphere/Icosphere.obj', do_normalize_model_file=False, svec=(R.marker_scale, R.marker_scale, R.marker_scale),M=M)
            R.bind_VBO(R.wired_mode)

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_position(window, xpos, ypos):

    if R.on_rotate == True:
        delta_angle = (pos_button_start - numpy.array((xpos, ypos))) * 1.0 * math.pi / W
        R.rotate_model((-delta_angle[1], -delta_angle[0], 0))

    if R.on_translate == True:
        delta_pos = (pos_button_start - numpy.array((xpos, ypos))) * 1.0/100
        R.translate_model((delta_pos[0], -delta_pos[1], 0))

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_scroll(window, xoffset, yoffset):

    if yoffset>0:R.scale_projection(1.10)
    else        :R.scale_projection(1.0 / 1.10)

    return
# ----------------------------------------------------------------------------------------------------------------------
def event_resize(window, W, H):
    R.resize_window(W,H)
    return
# ----------------------------------------------------------------------------------------------------------------------
def init_earth_ego():
    # press numkey "1"
    filename_obj = './images/ex_GL/earth/uv_sphere2.obj'

    textured = True
    rvec_model, tvec_model = (0, 0, 0), (0.0, 0.0, 0.0)
    M_obj = tools_pr_geom.compose_RT_mat(rvec_model, tvec_model, do_rodriges=False, do_flip=False, GL_style=True)
    eye = (0, 0, 0)
    target = (0, 1, 0)
    up = (0, 0, -1)
    return filename_obj, M_obj, textured, eye, target, up
# ----------------------------------------------------------------------------------------------------------------------
def init_box():

    filename_obj = './images/ex_GL/box/box_2.obj'
    textured = True
    rvec_model, tvec_model = (0, 0, 0),(0.0, 0.0, 0.0)
    M_obj = tools_pr_geom.compose_RT_mat(rvec_model,tvec_model,do_rodriges=False,do_flip=False, GL_style=True)
    eye = (0,10,0)
    target=(0,0,0)
    up=(0,0,+1)
    return filename_obj,M_obj,textured,eye,target,up
# ----------------------------------------------------------------------------------------------------------------------
def init_box_ego():
    filename_obj = './images/ex_GL/box/box_2.obj'
    textured = True
    rvec_model, tvec_model = (0, 0, 0),(0.0, 10.0, 0.0)
    M_obj = tools_pr_geom.compose_RT_mat(rvec_model,tvec_model,do_rodriges=False,do_flip=False, GL_style=True)
    eye = (0,0,0)
    target=(0,1,0)
    up=(0,0,-1)
    return filename_obj,M_obj,textured,eye,target,up
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/gl/'
W,H = 800,600
cam_fov_deg = 90
do_normalize_model_file = False
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #filename_obj,M_obj,textured,eye,target,up = init_box()
    filename_obj,M_obj,textured,eye,target,up = init_earth_ego()


    R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=W, H=H, do_normalize_model_file=do_normalize_model_file, projection_type='P',cam_fov_deg=cam_fov_deg,scale=(1, 1, 1),
                               M_obj=M_obj,textured=textured,
                               eye = eye,target=target,up=up)


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

