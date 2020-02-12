import numpy
import time
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5 import QtGui,QtCore
from scipy.spatial import Delaunay
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_GL3D
import detector_landmarks
import tools_video
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
capturing_devices = ['cam','mp4','image']
capturing_device = capturing_devices[2]
camera_W, camera_H = 640, 480
# ---------------------------------------------------------------------------------------------------------------------
class Thread(QThread):
    process_logic = pyqtSignal(QImage,QImage,QImage)
# ---------------------------------------------------------------------------------------------------------------------
    def get_image_3dmask(self, landmarks, Renderer, image_driver):
        del_triangles = Delaunay(landmarks).vertices
        D.draw_landmarks_v2(image_driver, landmarks, del_triangles)
        r_vec, t_vec = D.get_pose(image_driver, landmarks)

        image_3d = Renderer.get_image(r_vec, t_vec)
        clr = (255 * numpy.array(Renderer.bg_color)).astype(numpy.int)
        result = tools_image.blend_avg(image_driver, image_3d, clr, weight=0)
        return result
# ---------------------------------------------------------------------------------------------------------------------
    def get_image_landmarks_pose(self, landmarks, image_driver):
        del_triangles = Delaunay(landmarks).vertices
        D.draw_landmarks_v2(image_driver, landmarks, del_triangles)
        result = D.draw_landmarks(image_driver)
        r_vec, t_vec = D.get_pose(image_driver, landmarks)
        result = D.draw_annotation_box(result, r_vec, t_vec)
        return result

# ---------------------------------------------------------------------------------------------------------------------
    def run(self):

        Renderer = tools_GL3D.render_GL3D(filename_obj='./images/ex_GL/face/face.obj', W=camera_W, H=camera_H)

        if capturing_device == 'cam':
            cap = cv2.VideoCapture(0)
            self.cap.set(3, camera_W)
            self.cap.set(4, camera_H)
        elif capturing_device == 'mp4':
            cap = cv2.VideoCapture(filename_driver)

        image_3dmask = numpy.full((camera_H, camera_W, 3), 192, dtype=numpy.uint8)
        image_pose = numpy.full((camera_H, camera_W, 3), 64, dtype=numpy.uint8)

        cnt, start_time, fps = 0, time.time(), 0
        while True:

            if capturing_device == 'image':
                image_driver = image_driver_default.copy()
            else:
                ret, image_driver = cap.read()

            image_driver_display = image_driver.copy()

            if(cnt%3)>=0:
                landmarks = D.get_landmarks(image_driver)
                if D.are_landmarks_valid(landmarks):
                    image_3dmask = self.get_image_3dmask(landmarks,Renderer,image_driver)
                    image_pose = self.get_image_landmarks_pose(landmarks,image_driver)

            cnt += 1
            if time.time() > start_time: fps = cnt / (time.time() - start_time)
            cv2.putText(image_driver_display, '{0: 1.1f} {1}'.format(fps, ' fps'), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0), 1, cv2.LINE_AA)


            self.process_logic.emit(self.__to_QImage(image_driver_display),self.__to_QImage(image_3dmask),self.__to_QImage(image_pose))
# ---------------------------------------------------------------------------------------------------------------------
    def __to_QImage(self, numpy_image):
        Q_image = QtGui.QImage(numpy_image.data, numpy_image.shape[1], numpy_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        return Q_image
# ---------------------------------------------------------------------------------------------------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_UI()
# ---------------------------------------------------------------------------------------------------------------------
    @pyqtSlot(QImage,QImage,QImage)
    def update_UI(self, Qimage_camera, Qimage_3dmask, Qimage_pose):
        self.widget_camera.setPixmap(QPixmap.fromImage(Qimage_camera))
        self.widget_3dmask.setPixmap(QPixmap.fromImage(Qimage_3dmask))
        self.widget_pose.setPixmap(QPixmap.fromImage(Qimage_pose))
        return
# ---------------------------------------------------------------------------------------------------------------------
    def init_UI(self):
        self.setWindowTitle('DMS')
        self.setGeometry(1, 40, camera_W, camera_H)
        self.resize(1920, 1080)

        self.widget_camera = QLabel(self)
        self.widget_camera.move(0, 0)
        self.widget_camera.resize(camera_W, camera_H)

        self.widget_3dmask = QLabel(self)
        self.widget_3dmask.move(camera_W+10, 0)
        self.widget_3dmask.resize(camera_W, camera_H)

        self.widget_pose= QLabel(self)
        self.widget_pose.move(0, camera_H)
        self.widget_pose.resize(camera_W, camera_H)

        self.show()

        th = Thread(self)
        th.process_logic.connect(self.update_UI)
        th.start()

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    folder_in = './images/ex_faceswap/01/'
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    filename_driver = folder_in+list_filenames[0]

    image_driver_default = cv2.imread(filename_driver)
    image_driver_default = tools_image.smart_resize(image_driver_default, camera_H, camera_W)

    #filename_driver = './images/ex_DMS/JB_original.mp4'

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())