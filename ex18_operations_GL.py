import cv2
import math
import numpy
import pyrr
from scipy.linalg import polar
import tools_calibrate
import tools_render_CV
import detector_landmarks
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
def apply_matrix(M,X):
    if X.shape[1]==3:
        X4D = numpy.full((X.shape[0], 4), 1)
        X4D[:, :3] = X
    else:
        X4D = X

    Y1 = pyrr.matrix44.multiply(M, X4D.T).T
    Y2 = numpy.array([pyrr.matrix44.apply_to_vector(M.T, x) for x in X4D])

    return Y1
# ----------------------------------------------------------------------------------------------------------------------
def apply_rotation(rvec,X):
    R = pyrr.matrix44.create_from_eulers(rvec)
    Y = apply_matrix(R, X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def apply_translation(tvec,X):
    T = pyrr.matrix44.create_from_translation(tvec).T
    Y = apply_matrix(T,X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def apply_RT(rvec,tvec,X):
    R = pyrr.matrix44.create_from_eulers(rvec)
    T = pyrr.matrix44.create_from_translation(tvec).T
    M = pyrr.matrix44.multiply(T,R)
    Y = apply_matrix(M,X)
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def compose_RT_mat(rvec,tvec):

    R = pyrr.matrix44.create_from_eulers(rvec)
    T = pyrr.matrix44.create_from_translation(tvec).T
    M = pyrr.matrix44.multiply(T, R)

    return M
# ----------------------------------------------------------------------------------------------------------------------
def check_decompose():
    for i in range(1):
        T = numpy.eye(4)
        T[0,0]=3

        rvec, tvec = numpy.random.rand(3)*0.1,numpy.random.rand(3)
        M = pyrr.matrix44.multiply(compose_RT_mat(rvec, tvec),T)
        T, R, K = tools_render_CV.decompose_into_TRK(M)
        rvec2 = tools_calibrate.rotationMatrixToEulerAngles(R[:3,:3])
        tvec2 = T[:3,3]
        M2 = compose_RT_mat(rvec2, tvec2)

        print(rvec, tvec)
        print(rvec2, tvec2)
        print()
    return
# ----------------------------------------------------------------------------------------------------------------------
X = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)
filename_markers1 ='./images/ex_GL/face/markers_face.txt'
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_markers1)
# ----------------------------------------------------------------------------------------------------------------------
def xxx():
    image_actor = cv2.imread('./images/ex_faceswap/01/person1a.jpg')

    X = D.model_68_points
    points_2d = D.get_landmarks(image_actor)
    modelview = numpy.array([[0.26, 0.01, 0., -0.01], [-0.01, 0.23, 0., 0.04], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    scale_factor = 1
    fx, fy = image_actor.shape[1], image_actor.shape[0]

    camera_matrix = numpy.array([[fx, 0, fx / 2], [0, fy, fy / 2], [0, 0, 1]])
    P = numpy.eye(4)
    P[:3, :2] = camera_matrix[:, :2]
    P[:3, 3] = camera_matrix[:, 2] * scale_factor
    P /= scale_factor
    PRT = pyrr.matrix44.multiply(P, modelview)
    X4D = numpy.full((X.shape[0], 4), 1, dtype=numpy.float)
    X4D[:, :3] = X
    points_2d_check = pyrr.matrix44.multiply(PRT, X4D.T).T
    points_2d_check = (numpy.array(points_2d_check)[:, :2])

    for x in points_2d: cv2.circle(image_actor, (int(x[0]), int(x[1])), 4, (0, 128, 255), -1)
    for x in points_2d_check: cv2.circle(image_actor, (int(x[0]), int(x[1])), 4, (0, 32, 190), -1)
    cv2.imwrite('./images/output/fit_check_modelview.png', image_actor)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    check_decompose()