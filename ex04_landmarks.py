import numpy
import cv2
from scipy.spatial import Delaunay
import tools_GL
# ---------------------------------------------------------------------------------------------------------------------
import time
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
camera_W, camera_H = 640, 480
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
folder_in = 'D:/2/'
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def demo_live():
    R = tools_GL.render_GL(numpy.zeros((camera_H,camera_W,3),dtype=numpy.uint8))

    use_camera = True
    cnt, start_time, fps = 0, time.time(), 0

    cap = cv2.VideoCapture(0)
    cap.set(3, camera_W)
    cap.set(4, camera_H)
    while (True):
        if use_camera:
            ret, image = cap.read()
            L = D.get_landmarks(image)
            if D.are_landmarks_valid(L):
                del_triangles = Delaunay(L).vertices
                D.draw_landmarks_v2(image,L,del_triangles)
                result = D.draw_landmarks(image)
                r_vec, t_vec = D.get_pose(L)
                result = D.draw_annotation_box(result,r_vec, t_vec)
                #result = R.get_image(result)
                #result = R.morph_3D_mesh(camera_H,camera_W,result,r_vec, t_vec)

            else:
                result = image.copy()


        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        result = cv2.putText(result, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)
        if time.time() > start_time: fps = cnt / (time.time() - start_time)
        cv2.imshow('frame', result)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:break
        cnt+=1


    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    demo_live()






