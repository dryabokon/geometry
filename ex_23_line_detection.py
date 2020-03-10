import math
import cv2
import numpy
import cv2
import tools_soccer_field
import tools_GL3D
import tools_IO
from PIL import Image, ImageDraw, ImageOps
import pyrr
#----------------------------------------------------------------------------------------------------------------------
filename_in = './images/ex_lines/frame000269.jpg'
import tools_image
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_lines/'
folder_out = './images/output/'
filename_in = './images/ex_lines/frame000357.jpg'
#filename_in = './images/ex_lines/Image1.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in,do_debug=1):
    #tools_IO.remove_files(folder_out)
    image = cv2.imread(filename_in)
    result = SFP.process_view(image, do_debug)
    cv2.imwrite(folder_out + filename_in.split('/')[-1], result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02():
    filename_obj = './images/output/playground.obj'
    filename_texture = './images/output/playground.png'
    SFP.export_playground(filename_obj,filename_texture)

    r_vec, t_vec = [ -1.25, -0.47, -0.61], [-31.33,  -475.77,  -133.02]
    r_vec = numpy.array(r_vec)
    t_vec = numpy.array(t_vec)

    SFP.H,SFP.W = 900,1600
    image = SFP.draw_playground_pnp(r_vec,t_vec, w=2, R=6)
    cv2.imwrite(folder_out + 'GT_viewmat_CV.png', image)

    R = tools_GL3D.render_GL3D(filename_obj=filename_obj,filename_texture=filename_texture, W=SFP.W, H=SFP.H,do_normalize_model_file=False,is_visible=False,projection_type='P',scale=(1,1,1))
    image = R.get_image_perspective(r_vec, t_vec,do_debug=True)
    cv2.imwrite(folder_out + 'GT_viewmat_GL.png', image)


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_01(filename_in)
    #example_02()


