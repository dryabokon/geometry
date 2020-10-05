import numpy
import cv2
import tools_image
from skimage import exposure
import warnings
import tools_draw_numpy
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------------------------
import tools_Skeletone
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#filename_in = './images/ex_lines/frame000000_vanishing2.png'
filename_in = './images/ex_lines/frame000000.jpg'
# ----------------------------------------------------------------------------------------------------------------------
Ske = tools_Skeletone.Skelenonizer(folder_out)
W,H = 3*255,255
# ----------------------------------------------------------------------------------------------------------------------
def ex_grid_search(filename_in):
    import tools_CNN_view
    image = cv2.imread(filename_in)

    tensor = []
    for th1 in range(0, 255, 10):
        for th2 in range(th1 + 1, 255, 10):
            image_edges = cv2.Canny(image=image, threshold1=th1, threshold2=th2)
            image_edges = cv2.resize(image_edges, (W, H))
            image_edges = cv2.putText(image_edges, '{0} {1}'.format(th1, th2), (W // 2, H // 2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
            tensor.append(image_edges)

    tensor = numpy.transpose(numpy.array(tensor), (1, 2, 0))
    image = tools_CNN_view.tensor_gray_3D_to_image(tensor)
    cv2.imwrite(folder_out + 'res.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_edges(filename_in):
    image = cv2.imread(filename_in)
    image_edges = cv2.Canny(image=image, threshold1=20, threshold2=250)
    cv2.imwrite(folder_out + 'edges.png', image_edges)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in):

    sobel = numpy.zeros((8, 17), dtype=numpy.float32)
    sobel[:,  sobel.shape[1]//2:] = +1
    sobel[:, :sobel.shape[1]//2 ] = -1
    sobel = sobel/sobel.sum()


    image = cv2.imread(filename_in)
    gray = cv2.imread(filename_in,0)

    result = cv2.filter2D(gray, 0, sobel)
    result2 = numpy.maximum(255-result,result)
    result2 = exposure.adjust_gamma(result2, 6)

    image_bin = Ske.binarize(result2)
    image_ske = Ske.binarized_to_skeleton(image_bin)
    segemnts = Ske.skeleton_to_segments(image_ske)

    colors = tools_draw_numpy.get_colors(len(segemnts),shuffle=True)
    image_segm = tools_draw_numpy.draw_segments(tools_image.desaturate(image), segemnts, colors,w=1)

    cv2.imwrite(folder_out + 'filtered.png', result)
    cv2.imwrite(folder_out + 'filtered2.png', result2)
    cv2.imwrite(folder_out + 'ske.png', image_ske)
    cv2.imwrite(folder_out + 'segm.png', image_segm)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    example_01(filename_in)
