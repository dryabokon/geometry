import cv2
import numpy
import tools_IO
import processor_Slices
import tools_Skeletone
import tools_image
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_bin4/'
folder_out = './images/output_ex04/'
# ----------------------------------------------------------------------------------------------------------------------
P = processor_Slices.processor_Slices(folder_out)
S = tools_Skeletone.Skelenonizer(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def process_folder_images(folder_in, folder_out):

    tools_IO.remove_files(folder_out,create=True)
    filenames = tools_IO.get_filenames(folder_in,'*.jpg')

    for filename in filenames:
        P.process_file_granules(folder_in + filename, do_debug=False)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #process_folder_images(folder_in, folder_out)

    P.process_folder_json(folder_out, folder_out)



    

