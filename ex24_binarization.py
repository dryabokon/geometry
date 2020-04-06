import cv2
import numpy
import tools_IO
import processor_Slices
import tools_Skeletone
import tools_image
import tools_plot
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_bin/'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = processor_Slices.processor_Slices(folder_out)
S = tools_Skeletone.Skelenonizer()
# ----------------------------------------------------------------------------------------------------------------------
def process_avg(folder_in, folder_out):

    filenames = tools_IO.get_filenames(folder_in, '*.png')

    image = cv2.imread(folder_in + filenames[0])
    A = numpy.zeros_like(image,numpy.long)

    for filename in filenames:
        image = cv2.imread(folder_in + filename)
        A += image

    A = A / len(filenames)
    A/=A.max()
    A*=255

    cv2.imwrite(folder_out+'avg_dir.png',tools_image.hitmap2d_to_jet(A))

    return
# ----------------------------------------------------------------------------------------------------------------------
def mask_images(filename1_in,filename2_in):
    image1 = cv2.imread(filename1_in)
    image2 = cv2.imread(filename2_in)
    result = numpy.zeros_like(image2)

    for r in range(image1.shape[0]):
        for c in range(image1.shape[1]):
            col = image1[r,c]
            hsv = tools_image.bgr2hsv(col).reshape(3)
            l = image2[r, c,0]

            hsv[2]=l
            col = tools_image.hsv2bgr(hsv)
            result[r,c]=col


    cv2.imwrite(folder_out+'res.png',result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def process_folder(folder_in, folder_out):

    tools_IO.remove_files(folder_out)
    filenames = tools_IO.get_filenames(folder_in,'*.jpg')

    for filename in filenames[60:]:
        #P.process_file_ske(folder_in + filename, do_debug=True)
        #P.process_file_granules(folder_in + filename, do_debug=True)
        P.process_file_flow(folder_in + filename, do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #process_folder(folder_in, folder_out)
    #process_avg(folder_out, folder_out)

    mask_images(folder_out+'avg_dir2.png', folder_out+'avg2.png')

