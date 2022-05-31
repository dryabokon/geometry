import numpy
import cv2
import tools_image
import tools_IO
from CV import tools_Hough
from CV import tools_Skeletone
import tools_render_CV
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines(image, lines,color=(255,255,255),w=4,put_text=False):

    result = image.copy()
    H, W = image.shape[:2]
    for id,(x1,y1,x2,y2) in enumerate(lines):
        if len(numpy.array(color).shape)==1:
            clr = color
        else:
            clr = color[id].tolist()

        if numpy.any(numpy.isnan((x1,y1,x2,y2))): continue
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), clr, w)
        if put_text:
            x,y = int(x1+x2)//2, int(y1+y2)//2
            cv2.putText(result, '{0}'.format(id),(min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


    return result
# ----------------------------------------------------------------------------------------------------------------------
def example_01_lines_ski(filename_in, folder_out):

    angle_bound1 = 30
    angle_bound2 =-30

    image = cv2.imread(filename_in)
    H,W = image.shape[:2]

    skeleton = cv2.Canny(image, 20, 80)
    #skeleton = Ske.binarized_to_skeleton_ski(Ske.binarize(image))

    lines_vert, weights_vert = Hough.get_lines_ski(skeleton, max_count=5, min_weight=50,the_range=Hough.get_angle_range(angle_bound1, 20, 1))
    lines_horz, weights_horz = Hough.get_lines_ski(skeleton, max_count=5, min_weight=50,the_range =Hough.get_angle_range(angle_bound2,20, 1))

    lines_vert = numpy.array([tools_render_CV.line_box_intersection(line, (0, 0, W - 1, H - 1)) for line in lines_vert])
    lines_horz = numpy.array([tools_render_CV.line_box_intersection(line, (0, 0, W - 1, H - 1)) for line in lines_horz])

    image_lines = 1 * tools_image.desaturate(image.copy())
    image_lines = draw_lines(image_lines, lines_horz, color=(0, 128, 255), w=4)
    image_lines = draw_lines(image_lines, lines_vert, color=(0, 0, 255), w=4)

    alpha = 0.5
    gray3d = tools_image.desaturate(image)
    result = alpha * image_lines + (1 - alpha) * gray3d
    cv2.imwrite(folder_out + 'skeleton.png', skeleton)
    cv2.imwrite(folder_out + 'lines.png', result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_01_lines_cv(filename_in, folder_out):
    image = cv2.imread(filename_in)

    #preprocessed = Ske.binarized_to_skeleton_ski(Ske.binarize(image))
    preprocessed = cv2.Canny(image, 20, 80)
    lines,weights = Hough.get_lines_cv(preprocessed)

    result = tools_image.saturate(preprocessed.copy())

    for line,w in zip(reversed(lines),reversed(weights)):
        cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=(0,0,200), thickness=4)

    cv2.imwrite(folder_out + 'preprocessed.png', preprocessed)
    cv2.imwrite(folder_out + 'result.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#filename_in = './images/ex_lines/frame000159.jpg'
#filename_in = './images/ex_lines/frame000000_vanishing2.png'
#filename_in = './images/output/filtered.png'
filename_in = './images/ex_keypoints/_001644.jpg'
# ----------------------------------------------------------------------------------------------------------------------
Ske = tools_Skeletone.Skelenonizer(folder_out)
Hough = tools_Hough.Hough()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #example_01_lines_ski(filename_in,folder_out)
    example_01_lines_cv(filename_in, folder_out)
