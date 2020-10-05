import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import tools_ELSD
import tools_Skeletone
import tools_image

# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_ellipses/original.png'
# ----------------------------------------------------------------------------------------------------------------------
ELSD = tools_ELSD.ELSD(folder_out)
Ske = tools_Skeletone.Skelenonizer()
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipses(image, ellipses,color=(255,255,255),w=4,put_text=False):
    image_ellipses = image.copy()

    for e in ellipses:
        x1, y1, x2, y2, cx, cy, ax, bx, theta, ang_start, ang_end = e

        center = (int(cx), int(cy))
        angle_theta = int(theta * 180 / numpy.pi)
        arc_start = int(180 * ang_start / numpy.pi)
        arc_end = int(180 * ang_end / numpy.pi)

        cv2.ellipse(img=image_ellipses, center=center, axes=(int(ax), int(bx)), angle=angle_theta, startAngle=0,endAngle=360, color=color, thickness=1)

        if arc_start > arc_end:
            cv2.ellipse(img=image_ellipses, center=center, axes=(int(ax), int(bx)), angle=angle_theta, startAngle=0,endAngle=360 - arc_start, color=color, thickness=w)
            cv2.ellipse(img=image_ellipses, center=center, axes=(int(ax), int(bx)), angle=angle_theta, startAngle=0,endAngle=arc_end, color=color, thickness=w)
        else:
            cv2.ellipse(img=image_ellipses, center=center, axes=(int(ax), int(bx)), angle=angle_theta,startAngle=arc_start, endAngle=arc_end, color=color, thickness=w)

    return image_ellipses
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    image = cv2.imread(filename_in)

    #binarized = Ske.binarize(image)
    #edges = cv2.Canny(image=image, threshold1=100, threshold2=200)
    #morphed = tools_image.saturate(255-Ske.morph(edges, kernel_h=3, kernel_w=3, n_dilate=3, n_erode=0))
    #cv2.imwrite(folder_out + 'morphed.png', image)

    elipses = ELSD.extract_ellipses(image)
    result_image = draw_ellipses(image,elipses,color=(0,0,180))
    cv2.imwrite(folder_out+'result.png',result_image)


