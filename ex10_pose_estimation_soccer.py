import cv2
import numpy
import tools_render_CV
import tools_pr_geom
import tools_IO
import tools_soccer_GT_data
# ----------------------------------------------------------------------------------------------------------------------
soccer_data = tools_soccer_GT_data.Soccer_Field_GT_data()
# ----------------------------------------------------------------------------------------------------------------------
def draw_cube(H,W,rvec,tvec):
    image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
    aperture_x, aperture_y = 0.5, 0.5
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0], aperture_x, aperture_y)
    image = tools_render_CV.draw_cube_numpy(image, mat_camera, numpy.zeros(5), rvec, tvec)
    cv2.imwrite(folder_out + 'cube.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_colors(N,alpha_blend=None):
    colors = []
    for i in range(0, N):
        hue = int(255 * i / (N-1))
        color = cv2.cvtColor(numpy.array([hue, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0][0]
        if alpha_blend is not None:color = ((alpha_blend) * numpy.array((255, 255, 255)) + (1-alpha_blend) * numpy.array(color))
        colors.append((int(color[0]), int(color[1]), int(color[2])))

    return colors
# ----------------------------------------------------------------------------------------------------------------------
def draw_GT_lines(image, lines, ids, w=4,put_text=False):

    colors = get_colors(16,alpha_blend=0.5)
    H, W = image.shape[:2]
    result = image.copy()
    for line, id in zip(lines, ids):
        if line[0] is None or line[1] is None or line[2] is None or line[3] is None:continue
        cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), colors[id], w)
        if put_text:
            x,y = int(line[0]+line[2])//2, int(line[1]+line[3])//2
            cv2.putText(result, '{0}'.format(id),(min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 1, cv2.LINE_AA)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_playground_homography(image, homography,lmrks_GT, lines_GT, w=1, R=4):

    if homography is None: return image
    lines_GT_trans = cv2.perspectiveTransform(lines_GT.reshape((-1, 1, 2)), homography).reshape((-1, 4))
    lmrks_GT_trans = cv2.perspectiveTransform(lmrks_GT.reshape((-1, 1, 2)), homography).reshape((-1, 2))
    result = draw_GT_lines(image, lines_GT_trans, ids=numpy.arange(0, len(lines_GT), 1), w=w)
    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_playground_RT(r_vec, t_vec, W, H, w=1, R=4):
    fx, fy = W, H
    mat_camera = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
    image = numpy.full((H, W, 3), 64, numpy.uint8)
    if r_vec is None or t_vec is None: return image

    GT_Z=0
    landmarks_GT, lines_GT = soccer_data.get_GT()
    landmarks_GT3 = numpy.full((landmarks_GT.shape[0], 3), GT_Z, dtype=numpy.float)
    landmarks_GT3[:, :2] = landmarks_GT
    # landmarks_GT, jac = cv2.projectPoints(landmarks_GT3, r_vec, t_vec, mat_camera, numpy.zeros(4))
    landmarks_GT, jac = tools_pr_geom.project_points(landmarks_GT3, r_vec, t_vec, mat_camera, numpy.zeros(4))

    landmarks_GT = numpy.reshape(landmarks_GT, (-1, 2))


    lines_GT3 = numpy.full((2 * lines_GT.shape[0], 3), GT_Z, dtype=numpy.float)
    lines_GT3[:, :2] = lines_GT.reshape(-1, 2)
    # lines_GT, jac = cv2.projectPoints(lines_GT3, r_vec, t_vec, mat_camera, numpy.zeros(4))
    lines_GT, jac = tools_pr_geom.project_points(lines_GT3, r_vec, t_vec, mat_camera, numpy.zeros(4))
    lines_GT = numpy.reshape(lines_GT, (-1, 4))
    image = draw_GT_lines(image, lines_GT, ids=numpy.arange(0, 16, 1), w=w)

    return image
# ----------------------------------------------------------------------------------------------------------------------
def homography_to_RT(homography,W,H):
    fx, fy = W, H
    K = tools_pr_geom.compose_projection_mat_3x3(fx, fy)
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homography, K)

    return Rs, Ts, Ns
# ----------------------------------------------------------------------------------------------------------------------
def check_homography(H,W,rvec,tvec):

    image = numpy.full((H, W, 3), 64, dtype=numpy.uint8)
    landmarks_3d = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],
                                [-1, -1,  0], [-1, +1, +0], [+1, +1, +0], [+1, -1, +0]], dtype=numpy.float32)
    aperture_x, aperture_y = 0.5, 0.5
    mat_camera = tools_pr_geom.compose_projection_mat_3x3(image.shape[1], image.shape[0], aperture_x, aperture_y)
    landmarks_2d, jac = tools_pr_geom.project_points(landmarks_3d[[4,5,6,7]], rvec, tvec, mat_camera, numpy.zeros(5))

    idx_selected = [4,5,6,7]

    colors = tools_IO.get_colors(8)
    rvec2, tvec2, landmarks_2d_check = tools_pr_geom.fit_pnp(landmarks_3d[idx_selected], landmarks_2d, mat_camera)
    print('PNP R,T',rvec2, tvec2)
    for i, point in enumerate(landmarks_2d_check):cv2.circle(image, (point[0], point[1]), 5, colors[idx_selected[i]].tolist(), thickness=-1)
    cv2.imwrite(folder_out + 'check_pnp.png', image)


    landmarks_GT, lines_GT = soccer_data.get_GT()

    landmarks_GT[:,0]/=(2000/2)
    landmarks_GT[:,0]-=1
    landmarks_GT[:,1]/=(1500/2)
    landmarks_GT[:,1]-=1

    lines_GT[:, [0,2]] /= (2000 / 2)
    lines_GT[:, [0,2]] -= 1
    lines_GT[:, [1,3]] /= (1500 / 2)
    lines_GT[:, [1,3]] -= 1


    image = numpy.full((H, W, 3), 32, dtype=numpy.uint8)
    homography, result = tools_pr_geom.fit_homography(landmarks_GT[[19,18,0,1]],landmarks_2d)
    for i, point in enumerate(result.astype(int)):
        cv2.circle(image, (point[0], point[1]), 5, colors[idx_selected[i]].tolist(), thickness=-1)
    cv2.imwrite(folder_out + 'check_homography.png', image)
    playground = draw_playground_homography(image, homography, landmarks_GT, lines_GT, w=1, R=4)
    cv2.imwrite(folder_out + 'check_playground.png', playground)



    Rs, Ts, Ns = homography_to_RT(homography,W,H)
    camera_matrix_3x3 = tools_pr_geom.compose_projection_mat_3x3(W, H)
    points_3d = numpy.array(landmarks_GT[[19, 18, 0, 1]])
    points_3d = numpy.hstack((points_3d,numpy.zeros((4,1))))

    for (R,T,N) in zip(Rs, Ts, Ns):
        print('Decomp R,T\n', R, T,N)
        print()
        rotation = R
        translation = T
        normal = N
        points2d, jac= tools_pr_geom.project_points(points_3d, R, T, camera_matrix_3x3,numpy.zeros(5))


    #draw_playground_RT()

    return
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
H, W = 1080, 1920
rvec = numpy.array([0,0.3,0.2])
tvec = numpy.array([0, 0, -7])
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    draw_cube(H,W,rvec,tvec)
    check_homography(H,W,rvec, tvec)



