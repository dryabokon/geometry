import cv2
import numpy
from zhou_accv_2018 import p3l,p3p
import tools_render_CV
from CV import tools_pr_geom
import tools_IO
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
#soccer_data = tools_soccer_GT_data.Soccer_Field_GT_data()
# ----------------------------------------------------------------------------------------------------------------------
def example_p3l():

    # 3D lines are parameterized as pts and direction stacked into a tuple
    # instantiate a couple of points centered around the origin
    pts = 0.6 * (numpy.random.random((3, 3)) - 0.5)

    # generate normalized directions
    directions = 2 * (numpy.random.random((3, 3)) - 0.5)
    xxx = numpy.linalg.norm(directions, axis=1)
    xxx = xxx[:, None]
    directions /= xxx

    line_3d = (pts, directions)

    # Made up projective matrix
    K = numpy.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

    # A pose
    R_gt = numpy.array(
        [
            [0.89802142, -0.41500101, 0.14605372],
            [0.24509948, 0.7476071, 0.61725997],
            [-0.36535431, -0.51851499, 0.77308372],
        ]
    )
    t_gt = numpy.array([-0.0767557, 0.13917375, 1.9708239])

    # Sample to points from the line and project them to 2D
    pts_s = numpy.hstack((pts, pts + directions)).reshape((-1, 3))
    line_2d = (pts_s @ R_gt.T + t_gt) @ K.T


    # this variable is organized as (line, point, dim)
    line_2d = (line_2d / line_2d[:, -1, None])[:, :-1].reshape((-1, 2, 2))

    # Compute pose candidates.
    poses = p3l(line_2d=line_2d, line_3d=line_3d, K=K)

    # The error criteria for lines is to ensure that both 3D points and
    # direction, after transformation, are inside the plane formed by the
    # line projection. We start by computing the plane normals

    # line in 2D has two sampled points.
    line_2d_c = numpy.linalg.solve(K, numpy.vstack((line_2d.reshape((2 * 3, 2)).T, numpy.ones((1, 2 * 3))))).T
    line_2d_c = line_2d_c.reshape((3, 2, 3))

    # row wise cross product + normalization
    n_li = numpy.cross(line_2d_c[:, 0, :], line_2d_c[:, 1, :])
    n_li /= numpy.linalg.norm(n_li, axis=1)[:, None]

    # Print results
    print("R (ground truth):", R_gt, sep="\n")
    print("t (ground truth):", t_gt)
    print("Nr of possible poses:", len(poses))
    for i, pose in enumerate(poses):
        R, t = pose

        # The error criteria for lines is to ensure that both 3D points and
        # direction, after transformation, are inside the plane formed by the
        # line projection

        # pts
        pts_est = pts @ R.T + t
        err_pt = numpy.mean(numpy.abs(numpy.sum(pts_est * n_li, axis=1)))

        # directions
        dir_est = directions @ R.T
        err_dir = numpy.mean(numpy.arcsin(numpy.abs(numpy.sum(dir_est * n_li, axis=1))) * 180.0 / numpy.pi)

        print("Estimate -", i + 1)
        print("R (estimate):", R, sep="\n")
        print("t (estimate):", t)
        print("Mean pt distance from plane (m):", err_pt)
        print("Mean angle error from plane (°):", err_dir)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_p3p():
    # instantiate a couple of points centered around the origin

    pts = 0.6 * (numpy.random.random((3, 3)) - 0.5)

    # Made up projective matrix
    K = numpy.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]])

    # A pose
    R_gt = numpy.array(
        [
            [-0.48048015, 0.1391384, -0.86589799],
            [-0.0333282, -0.98951829, -0.14050899],
            [-0.8763721, -0.03865296, 0.48008113],
        ]
    )
    t_gt = numpy.array([-0.10266772, 0.25450789, 1.70391109])

    # Project points to 2D
    pts_2d = (pts @ R_gt.T + t_gt) @ K.T
    pts_2d = (pts_2d / pts_2d[:, -1, None])[:, :-1]

    # Compute pose candidates. the problem is not minimal so only one
    # will be provided
    poses = p3p(pts_2d=pts_2d, pts_3d=pts, K=K)

    # Print results
    print("R (ground truth):", R_gt, sep="\n")
    print("t (ground truth):", t_gt)

    print("Nr of possible poses:", len(poses))
    for i, pose in enumerate(poses):
        R, t = pose

        # Project points to 2D
        pts_2d_est = (pts @ R.T + t) @ K.T
        pts_2d_est = (pts_2d_est / pts_2d_est[:, -1, None])[:, :-1]
        err = numpy.mean(numpy.linalg.norm(pts_2d - pts_2d_est, axis=1))

        print("Estimate -", i + 1)
        print("R (estimate):", R, sep="\n")
        print("t (estimate):", t)
        print("Mean error (pixels):", err)
        return
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube(H,W,rvec,tvec):
    image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
    aperture_x, aperture_y = 0.5, 0.5
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0], aperture_x, aperture_y)
    image = tools_render_CV.draw_cube_numpy(image, mat_camera, numpy.zeros(5), rvec, tvec)
    cv2.imwrite(folder_out + 'cube.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def check_pnp(H,W,rvec,tvec):

    image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
    landmarks_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1], [-1, -1, +1], [-1, +1, +1], [+1, +1, +1],[+1, -1, +1]], dtype=numpy.float32)
    aperture_x, aperture_y = 0.5, 0.5
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0], aperture_x, aperture_y)
    landmarks_2d, jac = tools_pr_geom.project_points(landmarks_3d, rvec, tvec, mat_camera, numpy.zeros(5))

    noise = 0*numpy.random.rand(8,1,2)

    colors = tools_draw_numpy.get_colors(8)
    rvec2, tvec2, landmarks_2d_check = tools_pr_geom.fit_pnp(landmarks_3d, landmarks_2d+noise, mat_camera)

    for i, point in enumerate(landmarks_2d_check):
        cv2.circle(image, (point[0], point[1]), 5, colors[i].tolist(), thickness=-1)

    cv2.imwrite(folder_out + 'check_pnp.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_best_pose_p3p(poses,landmarks_2d,landmarks_3d,mat_camera):

    best_loss, best_pose = None,None
    for (R, t) in poses:
        landmarks_2d_check = (landmarks_3d @ R.T + tvec)
        landmarks_2d_check = landmarks_2d_check @ mat_camera.T
        norm = numpy.array([landmarks_2d_check[:, 2]]).T
        landmarks_2d_check = (landmarks_2d_check / norm)
        landmarks_2d_check = landmarks_2d_check[:,:2]

        loss = ((landmarks_2d_check-landmarks_2d)**2).mean()
        if best_loss is None or loss<best_loss:
            best_loss = loss
            best_pose = (R,t)

    return best_pose
# ----------------------------------------------------------------------------------------------------------------------
def check_p3p(H,W,rvec,tvec):

    image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
    landmarks_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1], [-1, -1, +1], [-1, +1, +1], [+1, +1, +1],[+1, -1, +1]], dtype=numpy.float32)
    aperture_x, aperture_y = 0.5, 0.5
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0], aperture_x, aperture_y)
    landmarks_2d, jac = tools_pr_geom.project_points(landmarks_3d, rvec, tvec, mat_camera, numpy.zeros(5))

    landmarks_2d = landmarks_2d.reshape((-1, 2))

    idx = [0, 1, 2]
    poses = p3p(landmarks_2d[idx], landmarks_3d[idx], mat_camera)
    (R,tvec2) = get_best_pose_p3p(poses,landmarks_2d,landmarks_3d,mat_camera)

    image = tools_render_CV.draw_cube_numpy(image, mat_camera, numpy.zeros(5), R, tvec2)
    cv2.imwrite(folder_out + 'check_p3p.png', image)

    return
# ----------------------------------------------------------------------------------------------------------------------
def check_p3l(H,W,rvec,tvec):
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(W, H, 0.5, 0.5)

    landmarks_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1], [-1, -1, +1], [-1, +1, +1], [+1, +1, +1],[+1, -1, +1]], dtype=numpy.float32)
    line_3d_a = numpy.hstack((landmarks_3d[3], landmarks_3d[0]))
    line_3d_b = numpy.hstack((landmarks_3d[0], landmarks_3d[1]))
    line_3d_c = numpy.hstack((landmarks_3d[1], landmarks_3d[2]))
    lines_3d = numpy.array([line_3d_a, line_3d_b, line_3d_c])

    # manual case
    lines_2d = 0.75 * numpy.array([[772, 495, 1020, 520], [1020, 520, 525, 700], [525, 700, 240, 665]])

    # automated case
    # landmarks_2d, jac = tools_pr_geom.project_points(landmarks_3d, rvec, tvec, mat_camera, numpy.zeros(5))
    # landmarks_2d = landmarks_2d.reshape((-1,2))
    # line_2da = numpy.concatenate([landmarks_2d[3],landmarks_2d[0]])
    # line_2db = numpy.concatenate([landmarks_2d[0],landmarks_2d[1]])
    # line_2dc = numpy.concatenate([landmarks_2d[1],landmarks_2d[2]])
    # lines_2d = numpy.array([line_2da,line_2db,line_2dc])

    poses = tools_pr_geom.fit_p3l(lines_3d,lines_2d,mat_camera)

    for option, (R,translation) in enumerate(poses):
        image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
        points_3d_proj, jac = tools_pr_geom.project_points(landmarks_3d.reshape((-1, 3)), R, translation, mat_camera,numpy.zeros(5))
        lines_3d_proj, jac = tools_pr_geom.project_points(lines_3d.reshape((-1,3)), R, translation, mat_camera, numpy.zeros(5))
        image = tools_draw_numpy.draw_cuboid(image, points_3d_proj, idx_mode=0)
        image = tools_draw_numpy.draw_lines(image, lines_3d_proj.reshape((-1,4)),color=(0, 128, 255),antialiasing=False,w=5)
        image = tools_draw_numpy.draw_lines(image, lines_2d, color=(0, 0, 255), w=1)


        cv2.imwrite(folder_out + 'check_p3l_%02d.png'%option, image)

    return
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
H, W = 1080, 1920
rvec = numpy.array([0,0.3,0.2])
tvec = numpy.array([0, 0, -7])
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    check_p3l(H,W,rvec, tvec)



