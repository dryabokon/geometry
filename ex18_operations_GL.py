import cv2
import math
import numpy
import pyrr
from scipy.linalg import polar
import tools_calibrate
import tools_render_CV
import detector_landmarks
import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
# ----------------------------------------------------------------------------------------------------------------------
X = numpy.array([[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1],[-1, -1, +1], [-1, +1, +1], [+1, +1, +1], [+1, -1, +1]],dtype = numpy.float32)
filename_markers1 ='./images/ex_GL/face/markers_face.txt'
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat',filename_markers1)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #check_decompose()

    X = numpy.array([10,0,0])
    rvec, tvec = (0,0,math.pi/2),(10,0,0)

    #Y = translate (rotate (X))
    Y1 = tools_pr_geom.apply_RT(rvec,tvec,X)[:3]
    Y2 = tools_pr_geom.apply_translation(tvec, tools_pr_geom.apply_rotation(rvec, X))[:3]
    Y3 = tools_pr_geom.apply_rotation(rvec, tools_pr_geom.apply_translation(tvec, X))[:3]
    print(Y1)
    print(Y2)
    print(Y3)

