import numpy
import cv2
import tools_soccer_field
import tools_IO
import tools_render_CV
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

    for filename in filenames[290:291]:
        SFP.process_view(folder_in+filename,do_debug=True)



    return
# ----------------------------------------------------------------------------------------------------------------------
#image = cv2.imread(folder_in+'frame000159.jpg')
#patches = tools_image.do_patch(image)
#for i,patch in enumerate(patches):cv2.imwrite(folder_out+'%03d.jpg'%i,patch)
#cv2.imwrite(folder_out+'res.png',tools_image.do_stitch(image.shape[0],image.shape[1],patches))
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    process_folder(folder_in,folder_out)

    #SFP.process_view(folder_in + 'frame002170.jpg', do_debug=True)
    #SFP.unit_test_01(cv2.imread(folder_in+'frame002170.jpg'))


