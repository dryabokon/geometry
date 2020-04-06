import numpy
import cv2
import tools_soccer_field
import tools_GL3D
import tools_pr_geom
import tools_IO
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_lines/'
folder_out = './images/output/'
filename_in = './images/ex_lines/frame000159.jpg'
#filename_in = './images/ex_lines/Image1.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in,do_debug=1):
    #tools_IO.remove_files(folder_out)
    image = cv2.imread(filename_in)
    result = SFP.process_view(image, do_debug)

    cv2.imwrite(folder_out + filename_in.split('/')[-1].split('.')[0]+'.png', result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02(folder_in,folder_out):
    tools_IO.remove_files(folder_out)
    SFP.process_folder(folder_in, folder_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_03():
    filename_obj = './images/output/playground.obj'
    filename_texture = './images/output/playground.png'
    SFP.export_playground(filename_obj,filename_texture)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_01(filename_in)
    #example_02(folder_in,folder_out)




