import cv2
import cupy
import numpy
import tools_image
from scipy import ndimage
import tools_filter
import tools_faceswap
import tools_cupy
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_input = 'images/ex_blend/'
    folder_output = 'images/output/'

    img1 = cv2.imread(folder_input + 'part1.png')
    img2 = cv2.imread(folder_input + 'part2.png')
    mask_original = 1 * (img2[:, :] == (0,0,0))
    mask_bin = mask_original.copy()
    mask_bin = numpy.array(numpy.min(mask_bin, axis=2), dtype=numpy.int)

    mask = tools_filter.sliding_2d(mask_bin, 10, 10, 'avg')
    mask = numpy.clip(2 * mask, 0, 1.0)

    mask = cupy.array(mask)
    img1 = cupy.array(img1)
    img2 = cupy.array(img2)
    result1 = tools_image.do_blend(img1, img2, mask)
    result2 = tools_cupy.do_blend(img1, img2,mask)

    result2 = cupy.asnumpy(result2)

    cv2.imwrite(folder_output + 'res1.png', result1)
    cv2.imwrite(folder_output + 'res2.png', result2)