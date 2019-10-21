import cv2
import numpy as numpy
import os
from scipy.spatial import Delaunay
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
import tools_calibrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from mayavi import mlab
import open3d as o3d
# ---------------------------------------------------------------------------------------------------------------------
def mesh():
    mesh = o3d.io.read_triangle_mesh("./images/ex_face_3d/knot.ply")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.1, 0.406, 0.6])
    o3d.visualization.draw_geometries([mesh])
# ---------------------------------------------------------------------------------------------------------------------
def my_write_triangle_mesh(filename_out, vertices,triangles):
    tools_IO.remove_file(filename_out)

    f_handle = open(filename_out, "a+")
    f_handle.write("ply\nformat ascii 1.0\ncomment Created by Open3D\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n" % (len(vertices), len(triangles)))
    for vertic in vertices:
        for v in vertic:f_handle.write("%f "% v)
        f_handle.write("\n")

    for triangle in triangles:
        f_handle.write("%d "%len(triangle))
        for t in triangle:f_handle.write("%d "% t)
        f_handle.write("\n")

    f_handle.close()
    return
# ---------------------------------------------------------------------------------------------------------------------
def converter(filename_in,filename_out):
    data = tools_IO.load_mat(filename_in, dtype=numpy.float32)

    N = data.shape[0] // 3
    ix = numpy.arange(0, N, 3)
    iy = numpy.arange(1, N + 1, 3)
    iz = numpy.arange(2, N + 2, 3)

    X, Y, Z = data[ix], data[iy], data[iz]
    xyz = numpy.vstack((numpy.vstack((X, Y)), Z)).T

    #del_triangles = Delaunay(numpy.array([X,Y,Z]).T).vertices


    pts = mlab.points3d(X, Y, Z)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh)
    i=0

    #my_write_triangle_mesh(filename_out, xyz, del_triangles)
    #for triangle in del_triangles:
    #    i0, i1, i2, i3 = triangle
    #    p0, p1, p2, p3 = xyz[i0, :], xyz[i1, :], xyz[i2, :], xyz[i3, :]

    return
# ---------------------------------------------------------------------------------------------------------------------
def render(filename_in):
    mesh = o3d.io.read_triangle_mesh(filename_in)


    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(xyz)
    #mesh = o3d.geometry.TriangleMesh()
    #vertices = numpy.asarray(mesh.vertices)
    #triangles = numpy.asarray(mesh.triangles)
    #mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #mesh.triangles = mesh0.triangles

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    converter('./images/ex_face_3d/2.txt',"./images/ex_face_3d/face.ply")

    #render("./images/ex_face_3d/knot2.ply")
    #render("./images/ex_face_3d/face.ply")
