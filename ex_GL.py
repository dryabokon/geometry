from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(200, 200, "hidden window", None, None)
# Make the window's context current
glfw.make_context_current(window)