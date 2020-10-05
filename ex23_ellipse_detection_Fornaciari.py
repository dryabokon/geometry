#https://www.researchgate.net/publication/263893437_A_fast_and_effective_ellipse_detector_for_embedded_vision_applications
#https://github.com/horiken4/ellipse-detection
import cv2
import math
import numpy
from scipy.special import ellipeinc
from scipy import optimize
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_Fornaciari
import tools_Skeletone
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
filename_in = './images/ex_ellipses/original.png'
# ----------------------------------------------------------------------------------------------------------------
Ske = tools_Skeletone.Skelenonizer()
F = tools_Fornaciari.Fornaciari(folder_out)
# ----------------------------------------------------------------------------------------------------------------
def match_segment_ellipse_slow(points,ellipse,tol=10):

    for point in points:
        dst = tools_render_CV.distance_point_ellipse(point,ellipse)
        if numpy.isnan(dst) or dst>tol:
            return False
    return True
# ----------------------------------------------------------------------------------------------------------------
def generate_ellipse_points(center_xy = (20,10),Rxy = (5.0,10.0),rotation_angle_deg=0,N=10):
    def generate_ellipse_angles(num, a, b):

        angles = 2 * numpy.pi * numpy.arange(num) / num
        if a < b:
            e = float((1.0 - a ** 2 / b ** 2) ** 0.5)
            tot_size = ellipeinc(2.0 * numpy.pi, e)
            arc_size = tot_size / num
            arcs = numpy.arange(num) * arc_size
            res = optimize.root(lambda x: (ellipeinc(x, e) - arcs), angles)
            angles = res.x
        elif b < a:
            e = float((1.0 - b ** 2 / a ** 2) ** 0.5)
            tot_size = ellipeinc(2.0 * numpy.pi, e)
            arc_size = tot_size / num
            arcs = numpy.arange(num) * arc_size
            res = optimize.root(lambda x: (ellipeinc(x, e) - arcs), angles)
            angles = numpy.pi / 2 + res.x
        else:
            numpy.arange(0, 2 * numpy.pi, 2 * numpy.pi / num)
        return angles

    rotation_angle = rotation_angle_deg*numpy.pi/180

    phi = generate_ellipse_angles(N,Rxy[0],Rxy[1])
    points = numpy.vstack((Rxy[0] * numpy.sin(phi), Rxy[1] * numpy.cos(phi))).T
    M = numpy.array([[numpy.cos(rotation_angle),-numpy.sin(rotation_angle),0] ,[numpy.sin(rotation_angle),numpy.cos(rotation_angle),0],[0,0,1]])
    points2 = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1,3)[:,:2]
    points2[:,0]+=center_xy[0]
    points2[:,1]+=center_xy[1]

    return points2.astype(numpy.int)
# ----------------------------------------------------------------------------------------------------------------
def draw_points(points):

    image = numpy.full((1.2*(points[:,1].max()), 1.2*(points[:,0].max()), 3), 32, dtype=numpy.uint8)
    image = F.draw_segments(image, [points], w=-1)
    cv2.imwrite(folder_out + 'ellipse_points.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------
def fit_ellipse_points(points):

    H,W = int(1.2 * points[:, 1].max()), int(1.2 * points[:, 0].max())

    image = numpy.full((H,W,3),32,dtype=numpy.uint8)
    ellipse = cv2.fitEllipse(points)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    rotation_angle = ellipse[2]

    cv2.ellipse(image, center, axes, rotation_angle, startAngle=0, endAngle=360, color=(0, 0, 190),thickness=4)
    image = F.draw_segments(image, [points],color=(255,255,255),w=-1)
    cv2.imwrite(folder_out + 'ellipse_fit.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------
def example_segment_detection():
    image = cv2.imread(filename_in)
    segments = F.extract_segments(image, min_size=30)

    image_debug = tools_image.desaturate(image.copy())
    colors = tools_IO.get_colors(len(segments), shuffle=True)
    image_debug = F.draw_segments(image_debug, segments, colors, w=4, put_text=True)
    cv2.imwrite(folder_out + 'result.png', image_debug)

    return
# ----------------------------------------------------------------------------------------------------------------
def example_extract_ellipses(do_debug=False):
    image = cv2.imread(filename_in)

    segments = F.extract_segments(image, min_size=30)
    F.segments_to_ellipse(image,segments,do_debug=do_debug)

    return
# ----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #tools_IO.remove_files(folder_out)
    #example_extract_ellipses(do_debug=True)

    points = generate_ellipse_points(center_xy=(400, 300), Rxy=(200, 100), rotation_angle_deg = 45, N=100)
    ellipse = cv2.fitEllipse(points)
    #d = distance_point_ellipse((370,420),ellipse,do_debug=True)

    tools_render_CV.ellipse_line_intersection(ellipse,(370,420,420,400),do_debug=True)
