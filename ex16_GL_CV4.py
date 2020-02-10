#https://github.com/totex/PyOpenGL_tutorials/blob/master/video_18_simple_light.py
import glfw
from OpenGL.GL import *
import numpy
import pyrr
from PIL import Image
from ObjLoader import *
import OpenGL.GL.shaders
# ----------------------------------------------------------------------------------------------------------------------
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
# ----------------------------------------------------------------------------------------------------------------------
def main():

    glfw.init()
    w_width, w_height = 800, 600
    window = glfw.create_window(w_width, w_height, "My OpenGL window", None, None)

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    obj = ObjLoader()
    obj.load_model('./images/ex_GL/box.obj')

    texture_offset = len(obj.vertex_index)*12
    normal_offset = (texture_offset + len(obj.texture_index)*8)

    frag_shader = """#version 330
                            in vec2 newTexture;
                            in vec3 fragNormal;
                            out vec4 outColor;
                            uniform sampler2D samplerTexture;
                            void main()
                            {
                                vec3 ambientLightIntensity = vec3(0.3f, 0.2f, 0.4f);
                                vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
                                vec3 sunLightDirection = normalize(vec3(-2.0f, -2.0f, 0.0f));
                                vec4 texel = texture(samplerTexture, newTexture);
                                vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
                                outColor = vec4(texel.rgb * lightIntensity, texel.a);
                            }"""



    vert_shader = """#version 330
                            in layout(location = 0) vec3 position;
                            in layout(location = 1) vec2 textureCoords;
                            in layout(location = 2) vec3 vertNormal;
                            uniform mat4 transform,view,model,projection,light;
                            out vec2 newTexture;
                            out vec3 fragNormal;
                            void main()
                            {
                                fragNormal = (light * vec4(vertNormal, 0.0f)).xyz;
                                gl_Position = projection * view * model * transform * vec4(position, 1.0f);
                                newTexture = textureCoords;
                            }"""

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, obj.model.itemsize * len(obj.model), obj.model, GL_STATIC_DRAW)

    #positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    #textures
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, obj.model.itemsize * 2, ctypes.c_void_p(texture_offset))
    glEnableVertexAttribArray(1)

    #normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(normal_offset))
    glEnableVertexAttribArray(2)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # load image
    numpyimage = numpy.full((255,255,3),255,dtype = numpy.uint8)
    image = Image.fromarray(numpyimage)
    image = Image.open("images/ex_GL/box.png")
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = numpy.array(list(flipped_image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    glClearColor(0.3, 0.3, 0.3, 0.5)
    glEnable(GL_DEPTH_TEST)

    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0]))
    projection = pyrr.matrix44.create_perspective_projection_matrix(65.0, w_width / w_height, 0.1, 100.0)
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")
    transform_loc = glGetUniformLocation(shader, "transform")
    light_loc = glGetUniformLocation(shader, "light")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    #glColor4f(20, 3, 2, 0.7)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time() )
        rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time() )

        glUniformMatrix4fv(transform_loc, 1, GL_FALSE, rot_y)
        glUniformMatrix4fv(light_loc, 1, GL_FALSE, rot_y)

        glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))

        glfw.swap_buffers(window)

    glfw.terminate()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()