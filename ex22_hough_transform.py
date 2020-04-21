import numpy
import cv2
import tools_image
import tools_IO
import tools_Hough
import tools_Skeletone
# ----------------------------------------------------------------------------------------------------------------------
S = tools_Skeletone.Skelenonizer()
Hough = tools_Hough.Hough()
# ----------------------------------------------------------------------------------------------------------------------
def example_01_ski(filename_in,folder_out):

    image = cv2.imread(filename_in)

    binarized = S.binarize(image)
    skeleton = S.binarized_to_skeleton2(binarized)
    the_range = numpy.arange(60 * numpy.pi / 180, 80 * numpy.pi / 180, (numpy.pi / 1800))

    #lines,weights = Hough.get_lines_ski(skeleton,the_range,min_weight=100,max_count=10)
    lines,weights = Hough.get_lines_ski_segments(skeleton,the_range,min_weight=10,max_count=100)

    colors = tools_IO.get_colors(256)
    result = tools_image.saturate(image)
    for line,w in zip(reversed(lines),reversed(weights)):
        cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=colors[int(w)].tolist(), thickness=4)

    cv2.imwrite(folder_out + 'skeleton.png', skeleton)
    cv2.imwrite(folder_out + 'result.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_01_cv(filename_in,folder_out):

    preprocessed = Hough.preprocess(cv2.imread(filename_in), min_length=100)
    lines,weights = Hough.get_lines_cv(preprocessed,min_length=50)

    colors = tools_IO.get_colors(256)
    result = tools_image.saturate(preprocessed.copy())
    for line,w in zip(reversed(lines),reversed(weights)):
        cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=colors[int(w)].tolist(), thickness=4)

    cv2.imwrite(folder_out + 'preprocessed.png', preprocessed)
    cv2.imwrite(folder_out + 'result.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#filename_in = './images/ex_hough/image1.png'
filename_in = './images/ex_lines/frame000159.jpg'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_01_ski(filename_in,folder_out)


