import cv2
import numpy
import cv2.aruco as aruco
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
filename_in = 'images/ex_aruco/01.jpg'
filename_out = 'images/output/aruco_out.jpg'
USE_CAMERA = False


# ---------------------------------------------------------------------------------------------------------------------
def demo_aruco(camera_matrix, dist):

    marker_length  = 0.5

    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
    else:
        cap = []
        frame = cv2.imread(filename_in)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)

    while (True):
        if USE_CAMERA:
            ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = tools_image.desaturate(frame)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            gray_rgb = aruco.drawDetectedMarkers(gray_rgb, corners)
            for each in corners:
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(each, marker_length, camera_matrix, dist)
                aruco.drawAxis(gray_rgb, camera_matrix, dist, rvec[0], tvec[0], marker_length/2)

        cv2.imshow('frame', gray_rgb)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if USE_CAMERA:
        cap.release()

    cv2.destroyAllWindows()
    cv2.imwrite(filename_out,gray_rgb)

    return
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    width, height = 1200,800

    camera_matrix = numpy.array([[height, 0, width], [0, height, width], [0, 0, 1]]).astype(numpy.float64)
    dist = numpy.zeros((1, 5))
    demo_aruco(camera_matrix, dist)
