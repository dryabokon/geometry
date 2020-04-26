import numpy
import cv2
import tools_draw_numpy
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
color_blue = (255, 128, 0)
color_red = (20, 0, 255)
color_gray = (180,180,180)
# ----------------------------------------------------------------------------------------------------------------------
def check_intersection(segmxy1,segmxy2,fixed_endpooints = True):

    image = numpy.full((1080, 1920, 3), 64, dtype=numpy.uint8)
    cv2.line(image, (segmxy1[0], segmxy1[1]), (segmxy1[2], segmxy1[3]), color_blue, thickness=2)
    cv2.line(image, (segmxy2[0], segmxy2[1]), (segmxy2[2], segmxy2[3]), color_red, thickness=2)
    p1,p2, dist = tools_render_CV.distance_between_lines(segmxy1, segmxy2,clampAll=fixed_endpooints)
    print('distance = ', dist)

    if (p1 is not None) and (p2 is not None):
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        if fixed_endpooints:
            cv2.line(image, (p1[0],p1[1]), (p2[0],p2[1]), color_gray, thickness=2)
        else:
            cv2.circle(image, (p1[0], p1[1]), 10, color_gray, thickness=2)


    return image
# ----------------------------------------------------------------------------------------------------------------------
def example_01_check_intersection():
    segmxy1 = (100, 100, 1000, 100)
    segmxy2 = (200, 200, 400, 200)
    fixed_endpooints = True
    result = check_intersection(segmxy1, segmxy2, fixed_endpooints=fixed_endpooints)
    cv2.imwrite(folder_out + 'ex01_parallel.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02_check_intersection():
    segmxy1 = (100, 100, 1000, 100)
    segmxy2 = (200, 200, 250, 300)
    fixed_endpooints = True
    result = check_intersection(segmxy1, segmxy2, fixed_endpooints=fixed_endpooints)
    cv2.imwrite(folder_out + 'ex02_no_intersection.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_03_check_intersection():
    segmxy1 = (100, 100, 1000, 100)
    segmxy2 = (200, 200, 250, 300)
    fixed_endpooints = False
    result = check_intersection(segmxy1, segmxy2, fixed_endpooints=fixed_endpooints)
    cv2.imwrite(folder_out + 'ex03_intersection_free_endpooints.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_04_distance_segment_to_line():
    #line = (10, 100, 1000, 100)
    #segm = (200, 200, 80, 90)

    segm = [1920, 281, 0, 623]
    lines = [[-1369.14, 867.18, 1584.37, 341.08],
             [-1325.43, 1028.71, 1617.43, 446.],
             [-1259.76, 1240.39, 1667.99, 585.96],
             [-1178.02, 1475.58, 1732.87, 749.81],
             [-1510.85, 167.71, 1485.04, 324.72],
             [-1521.51, 246.43, 1470.81, 460.92],
             [-1556.03, 509.86, 1430.16, 797.39],
             [-1598.1, 775.79, 1382.62, 1115.4]]

    for line in numpy.array(lines).astype(int):
        d = tools_render_CV.distance_segment_to_line(segm, line)
        print(d)

        image = numpy.full((1080, 1920, 3), 64, dtype=numpy.uint8)
        cv2.line(image, (segm[0], segm[1]), (segm[2], segm[3]), color_blue, thickness=2)
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color_red, thickness=2)
        cv2.imwrite(folder_out+'04_dist_segment_to_line.png',image)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_05_trim_line_by_box():
    lines = numpy.array([[2147, 240, 15016, 924],
             [15016, 924, 52375, 9775],
             [52375, 9775, -2413, 1053],
             [-2413, 1053, 2147, 240],
             [1634, 332, 3056, 419],
             [3056, 419, 638, 1022],
             [638, 1022, -1085, 816],
             [1072, 432, 1538, 465],
             [1538, 465, 370, 695],
             [370, 695, -132, 646],
             [16340, 1238, 11775, 957],
             [11775, 957, 16125, 2873],
             [16125, 2873, 30479, 4588],
             [18084, 1651, 15934, 1497],
             [15934, 1497, 19708, 2557],
             [19708, 2557, 23412, 2914]])

    #line = (30, 150,1000, 150)
    pad =10
    box = (0+pad, 0+pad, 1920-pad, 1080-pad)

    for l,line in enumerate(lines):
        line_trimmed = tools_render_CV.trim_line_by_box(line, box)
        image = numpy.full((1080, 1920, 3), 64, dtype=numpy.uint8)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color_blue, thickness=2)
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color_gray, thickness=4)

        if not numpy.any(numpy.isnan(line_trimmed)):
            cv2.line(image, (line_trimmed[0], line_trimmed[1]), (line_trimmed[2], line_trimmed[3]), color_red, thickness=2)

        cv2.imwrite(folder_out + '05_trim_%02d.png'%l, image)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_06_ratio_skewed_rect():
    vert1 = numpy.array([-1369.14, 867.18, 1584.37, 341.08],dtype=int)
    vert2 = numpy.array([-1178.02, 1475.58, 1732.87,749.81],dtype=int)
    horz1 = numpy.array([-1521.51, 246.43, 1470.81, 460.92],dtype=int)
    horz2 = numpy.array([-1556.03, 509.86, 1430.16, 797.39],dtype=int)

    ratio = tools_render_CV.get_ratio_4_lines(vert1, horz1, vert2, horz2)

    p11, p12, dist = tools_render_CV.distance_between_lines(vert1, horz1, clampAll=False)
    p21, p22, dist = tools_render_CV.distance_between_lines(vert1, horz2, clampAll=False)
    p31, p32, dist = tools_render_CV.distance_between_lines(vert2, horz1, clampAll=False)
    p41, p42, dist = tools_render_CV.distance_between_lines(vert2, horz2, clampAll=False)
    rect = (p11[:2], p21[:2], p31[:2], p41[:2])

    image = numpy.full((1080, 1920, 3), 64, dtype=numpy.uint8)
    cv2.line(image, (vert1[0], vert1[1]), (vert1[2], vert1[3]), color_red, thickness=4)
    cv2.line(image, (vert2[0], vert2[1]), (vert2[2], vert2[3]), color_red, thickness=4)
    cv2.line(image, (horz1[0], horz1[1]), (horz1[2], horz1[3]), color_blue, thickness=4)
    cv2.line(image, (horz2[0], horz2[1]), (horz2[2], horz2[3]), color_blue, thickness=4)

    image = tools_draw_numpy.draw_convex_hull(image, rect, color=color_gray, transperency=0.25)
    cv2.imwrite(folder_out + '06_ratio.png', image)
    print(ratio)

    return ratio
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    example_04_distance_segment_to_line()


