import cv2
import numpy
import os
import argparse
import tools_image
from faceai.Detection import FacesDetection
from faceai.Alignment import LandmarksDetection
from faceai.ThrDFace import ThreeDimRestructure
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_landmark
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
default_filename_in  = './images/ex_faceswap/01/personC-1.jpg'
default_filename_in2 = './images/ex_faceswap/01/personD-1.jpg'
default_folder_out  = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
input_path = './images/ex_faceswap/01/personD-1.jpg'
# ---------------------------------------------------------------------------------------------------------------------
def afine(r,t):
  A = numpy.eye(4)
  A[:3,:3] = r
  A[:2,3] = t
  return A
# ---------------------------------------------------------------------------------------------------------------------
def transform(H, v):
    vv = np.ones((data.shape[0], 4))
    vv[:, :3] = v
    vv = np.dot(H, vv.T).T[:, :3]
    return vv
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    facedetector = FacesDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.loadModel(detection_speed='fast', min_face_size=12)


    img, infs = facedetector.detectFacesFromImage(input_image=input_path, box_mark=False)
    dets = []
    for inf in infs:dets.append(inf["detection_details"])

    landsdetector = ThreeDimRestructure()
    landsdetector.setModelTypeAsPRNet()
    landsdetector.loadModel()

    img_3d = landsdetector.restructure3DFaceFromImage(img, dets=dets, depth=True, pose=True)

    data = img_3d[0]['img_3d_inf']['vertices']

    scale, rotation, translation = img_3d[0]['img_3d_inf']['afine']
    A = afine(rotation, translation)

    zero_position = transform(np.linalg.inv(A), data)

    x0, y0 = zero_position[:, 0].min(), zero_position[:, 1].min()
    h0, w0 = int((zero_position[:, 1].max() - y0)), int((zero_position[:, 0].max() - x0))
    matrix_0 = np.zeros((h0, w0, 3))

    # compare aligned and not aligned face vertices
    x, y = data[:, 0].min(), data[:, 1].min()
    h, w = int((data[:, 1].max() - y)), int((data[:, 0].max() - x))
    matrix = np.zeros((h, w, 3))

    plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.scatter(zero_position[:, 0] - x0, zero_position[:, 1] - y0, s=0.2, c='r')
    plt.imshow(matrix_0)

    plt.subplot(122)
    plt.scatter(data[:, 0] - x, data[:, 1] - y, s=0.2, c='r')
    plt.imshow(matrix)

    plt.show()