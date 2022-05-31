import struct
import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
from CV import tools_calibrator
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
Calibrator = tools_calibrator.Calibrator()
W, H = 800, 600
col_empty = (32,32,32)
color_fill = tools_draw_numpy.color_gold
color_edge = tools_draw_numpy.color_red
w = 2
alpha = 0.5
# ----------------------------------------------------------------------------------------------------------------------
def get_points():
    points_3d = Calibrator.construct_cuboid_v0((-1, -1, -1, +1, +1, +1))
    rvec, tvec, fov = numpy.array([0.13, 0.2, 0]), numpy.array([+2.0, 0, +15]), 0.50
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H, fov, fov)
    points = tools_pr_geom.project_points(points_3d,rvec, tvec,camera_matrix_3x3,numpy.zeros(5))[0]
    return points
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out,'*.png')
    empty = numpy.full((H, W, 3), col_empty, dtype=numpy.uint8)
    points = get_points()

    cv2.imwrite(folder_out + 'points.png',tools_draw_numpy.draw_points(empty, points,color=color_edge,w=w,transperency=alpha))
    cv2.imwrite(folder_out + 'line.png',tools_draw_numpy.draw_line(empty, points[0][1], points[0][0], points[1][1], points[1][0], color_bgr=color_edge, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'lines.png',tools_draw_numpy.draw_lines(empty, points.reshape((-1,4)),color=color_edge,w=w,transperency=alpha))

    cv2.imwrite(folder_out + 'rect.png',tools_draw_numpy.draw_rect(empty, points[0][0], points[0][1], points[3][0], points[3][1], color=color_fill, w=w, alpha_transp=alpha, label='label 1'))

    cv2.imwrite(folder_out + 'ellipse.png', tools_draw_numpy.draw_ellipse(empty,(points[[0,3]].flatten()),color=color_fill,col_edge=color_edge,transperency=alpha))
    cv2.imwrite(folder_out + 'ellipse0.png',tools_draw_numpy.draw_ellipse0(empty, points[0][0], points[0][1], 100, 200, color_brg=color_fill, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'circle.png',tools_draw_numpy.draw_circle(empty, numpy.mean(points[:,1]), numpy.mean(points[:,0]), (points[3][0]-points[0][0])/2, color_brg=color_fill, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'circle_aa.png',tools_draw_numpy.draw_circle_aa(empty, numpy.mean(points[:,1]), numpy.mean(points[:,0]), (points[3][0]-points[0][0])/2, color_brg=color_fill, clr_bg=col_empty, alpha_transp=alpha))

    cv2.imwrite(folder_out + 'convex_cv.png',tools_draw_numpy.draw_convex_hull_cv(empty, points, color=color_fill, transperency=alpha))
    cv2.imwrite(folder_out + 'convex_PIL.png',tools_draw_numpy.draw_convex_hull(empty, points, color=color_fill, transperency=alpha))

    cv2.imwrite(folder_out + 'cuboid.png', tools_draw_numpy.draw_cuboid(empty, points, color=color_fill, w=w))
    cv2.imwrite(folder_out + 'contours.png',tools_draw_numpy.draw_contours(empty, points[[0,1,3,2]], color=color_fill, w=w,transperency=alpha))

    cv2.imwrite(folder_out + 'text.png',tools_draw_numpy.draw_text(empty,u'\u00B0',(100,100), color_fg=(255,255,0)))

