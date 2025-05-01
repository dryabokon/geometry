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
def text_align():
    image = numpy.full((H, W, 3), col_empty, dtype=numpy.uint8)
    image = tools_draw_numpy.draw_rect(image, 10,  10, 500, 300, color=color_fill, w=w, alpha_transp=alpha)
    image = tools_draw_numpy.draw_rect(image, 10, 300, 500, 500, color=color_fill, w=w, alpha_transp=alpha)
    image = tools_draw_numpy.draw_text(image, '▕ hor left vert top'   , (501,  10), color_fg=(0, 255, 200),clr_bg=(64,64,64),hor_align='left',vert_align='top')
    image = tools_draw_numpy.draw_text(image, '▕ hor center vert center', (501, 300), color_fg=(0, 255, 200),clr_bg=(64,64,64),hor_align='center',vert_align='center')
    image = tools_draw_numpy.draw_text(image, '▕ hor right vert bottom', (501, 500), color_fg=(0, 255, 200),clr_bg=(64,64,64),hor_align='right',vert_align='botttom')
    cv2.imwrite(folder_out + 'test.png',image)
    return ''
# ----------------------------------------------------------------------------------------------------------------------
def ex_draw():
    empty = numpy.full((H, W, 3), col_empty, dtype=numpy.uint8)
    #empty = cv2.imread('./images/ex_aruco/01.jpg')
    points = get_points()
    points += numpy.random.random(points.shape)

    cv2.imwrite(folder_out + 'points.png',tools_draw_numpy.draw_points(empty, points,color=color_edge,w=w,transperency=alpha))

    cv2.imwrite(folder_out + 'line.png',tools_draw_numpy.draw_line(empty, points[0][1], points[0][0], points[1][1], points[1][0], color_bgr=color_edge, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'lines.png',tools_draw_numpy.draw_lines(empty, points.reshape((-1,4)),color=color_edge,w=w,transperency=alpha))

    cv2.imwrite(folder_out + 'rect.png',tools_draw_numpy.draw_rect(empty, points[0][0], points[0][1], points[3][0], points[3][1], color=color_fill, w=w, alpha_transp=alpha, label='label 1'))
    #cv2.imwrite(folder_out + 'rect.png',tools_draw_numpy.draw_rect(empty, points[0][0], points[0][1], points[3][0], points[3][1], color=color_fill, w=-1, alpha_transp=0))

    cv2.imwrite(folder_out + 'ellipse.png', tools_draw_numpy.draw_simple_ellipse(empty,(points[[0,3]].flatten()),color=color_fill,col_edge=color_edge,transperency=alpha))
    cv2.imwrite(folder_out + 'ellipse0.png',tools_draw_numpy.draw_simple_ellipse0(empty, points[0][0], points[0][1], 100, 200, color_brg=color_fill, alpha_transp=alpha))

    cv2.imwrite(folder_out + 'circle.png',tools_draw_numpy.draw_circle(empty, numpy.mean(points[:,1]), numpy.mean(points[:,0]), (points[3][0]-points[0][0])/2, color_brg=color_fill, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'circle_aa.png',tools_draw_numpy.draw_circle_aa(empty, numpy.mean(points[:,1]), numpy.mean(points[:,0]), (points[3][0]-points[0][0])/2, color_brg=color_fill, clr_bg=col_empty, alpha_transp=alpha))
    cv2.imwrite(folder_out + 'circles.png',tools_draw_numpy.draw_circles_aa(empty, [(numpy.mean(points[:, 1]), numpy.mean(points[:, 0]))], colors=color_edge, w=(points[3][0]-points[0][0])/2, clr_bg=col_empty,transperency=alpha))

    cv2.imwrite(folder_out + 'convex_cv.png',tools_draw_numpy.draw_convex_hull_cv(empty, points, color=color_fill, transperency=alpha))
    cv2.imwrite(folder_out + 'convex_PIL.png',tools_draw_numpy.draw_convex_hull(empty, points, color=color_fill, transperency=alpha))

    cv2.imwrite(folder_out + 'cuboid.png', tools_draw_numpy.draw_cuboid(empty, points, color=color_fill, w=w))
    cv2.imwrite(folder_out + 'contours.png',tools_draw_numpy.draw_contours_cv(empty, points[[0,1,3,2]], color=color_fill, w=w,transperency=alpha))

    cv2.imwrite(folder_out + 'text.png',tools_draw_numpy.draw_text(empty,'This is \na sample text '+u'\u00B0',(100,100), color_fg=(255,255,0)))
    cv2.imwrite(folder_out + 'text_fast.png',tools_draw_numpy.draw_text_fast(empty, 'This is \na sample text ' + u'\u00B0', (100, 100),color_fg=(255, 255, 0)))


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out,'*.png')
    #ex_draw()
    empty = numpy.full((H, W, 3), col_empty, dtype=numpy.uint8)
    image = tools_draw_numpy.draw_rect(empty, 100, 100, 300, 200, (0,0,200), w=1, alpha_transp=0.8, font_size=32, label='xxx')
    image_fast = tools_draw_numpy.draw_rect_fast(empty, 100, 100, 300, 200, (0,50,200), w=1,font_size=32, label='xxx')

    cv2.imwrite(folder_out + 'draw_rect.png',image)
    cv2.imwrite(folder_out + 'draw_rect_fast.png', image_fast)

