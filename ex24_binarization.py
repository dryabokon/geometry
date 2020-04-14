import cv2
import numpy
import tools_IO
import processor_Slices
import tools_Skeletone
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_bin/'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = processor_Slices.processor_Slices(folder_out)
S = tools_Skeletone.Skelenonizer()
# ----------------------------------------------------------------------------------------------------------------------
def process_folder(folder_in, folder_out):

    tools_IO.remove_files(folder_out)
    filenames = tools_IO.get_filenames(folder_in,'*.jpg')

    for filename in filenames:
        P.process_file_granules(folder_in + filename, do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    process_folder(folder_in, folder_out)

