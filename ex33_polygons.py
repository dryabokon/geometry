import cv2
import numpy
import shapely.geometry as geom
from scipy.spatial import ConvexHull
# ----------------------------------------------------------------------------------------------------------------------
import tools_polygons_i
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
N = 50
image = numpy.full((600, 800, 3), 255, dtype=numpy.uint8)
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def get_shape(N,center,radius):
    x = center[0] + radius*numpy.array([numpy.sin(a/(N-1)*2*numpy.pi) for a in range(N)])
    y = center[1] + radius*numpy.array([numpy.cos(a/(N-1)*2*numpy.pi) for a in range(N)])
    x+= 0.9*radius*numpy.random.random(N)
    y+= 0.9*radius*numpy.random.random(N)
    points = numpy.concatenate((x.reshape((-1,1)),y.reshape((-1,1))),axis=1)
    hull = ConvexHull(numpy.array(points))
    cntrs = numpy.array(points)[hull.vertices]
    points = numpy.array([(point[0], point[1]) for point in cntrs])
    return points
# ----------------------------------------------------------------------------------------------------------------------
def interpolate(points1,points2,color1,color2):

    p1s = geom.Polygon(points1)
    p2s = geom.Polygon(points2)
    I = tools_polygons_i.PolygonInterpolator(p1=p1s, p2=p2s)

    X, Y, C0, C1, C2 = [], [], [], [], []
    for pair in I.tuple_pairs:
        X.append(numpy.linspace(pair[0][0], pair[1][0], N))
        Y.append(numpy.linspace(pair[0][1], pair[1][1], N))

    C0 = numpy.linspace(color1[0], color2[0], N).reshape((-1, 1))
    C1 = numpy.linspace(color1[1], color2[1], N).reshape((-1, 1))
    C2 = numpy.linspace(color1[2], color2[2], N).reshape((-1, 1))

    X = numpy.array(X).T
    Y = numpy.array(Y).T
    C = numpy.concatenate([C0, C1, C2], axis=1).astype(numpy.uint8)

    return X, Y, C
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    N1 = 12
    N2 = 15
    c1 = (400,300)
    c2 = (430,220)
    r1 = 100
    r2 = 150
    color1 = (0, 10, 255)
    color2 = (255,128,15)

    points1 = get_shape(N1,c1,r1)
    points2 = get_shape(N2,c2,r2)

    cv2.imwrite(folder_out+'start.png',tools_draw_numpy.draw_contours(image, points1, color=color1,transperency=0.9))
    cv2.imwrite(folder_out+'stop.png' ,tools_draw_numpy.draw_contours(image, points2, color=color2,transperency=0.9))

    X, Y, C = interpolate(points1, points2, color1, color2)

    for i in range(X.shape[0]):
        p = numpy.concatenate((X[i].reshape((-1,1)),Y[i].reshape((-1,1))),axis=1)
        cv2.imwrite(folder_out+'%03d.png'%i,tools_draw_numpy.draw_contours(image, p, color=C[i],transperency=0.9))





