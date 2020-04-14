import numpy
import cv2
import tools_soccer_field
import tools_GL3D
import tools_pr_geom
import tools_IO
import tools_Skeletone
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_lines/'
folder_out = './images/output/'
filename_in = './images/ex_lines/frame000159.jpg'
#filename_in = './images/ex_lines/Image1.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def process_folder(folder_in, folder_out):
    tools_IO.remove_files(folder_out, create=True)
    filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    for filename in filenames[2:3]:
        SFP.process_view(folder_in+filename,do_debug=True)


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_03():
    filename_obj = './images/output/playground.obj'
    filename_texture = './images/output/playground.png'
    SFP.export_playground(filename_obj,filename_texture)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    process_folder(folder_in,folder_out)







