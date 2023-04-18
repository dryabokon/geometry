import numpy
import cv2
# --------------------------------------------------------------------------------------------------------------------------
import tools_optical_flow
import tools_IO
# --------------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
folder_in = './images/ex_optical_flow/02/'
# --------------------------------------------------------------------------------------------------------------------------
capturing_device = 'cam'
# --------------------------------------------------------------------------------------------------------------------------
# capturing_device = 'mp4'
# filename_video = './images/ex_optical_flow/03.mp4'
# --------------------------------------------------------------------------------------------------------------------------
def exampl_GUI_loop():

    window_name = 'tracker UI'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    #face_2d = numpy.array([[1560, 262], [1811, 234], [1768, 736], [1532, 665]])
    face_2d = None

    if capturing_device == 'cam':
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cap.set(3, 720)
        cap.set(4, 640)
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
    else:
        cap = cv2.VideoCapture(filename_video)
        ret, image = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cv2.resizeWindow(window_name, image.shape[1]//2, image.shape[0]//2)

    OF = tools_optical_flow.OpticalFlow_LucasKanade(image, face_2d=face_2d, folder_out=folder_out)

    should_be_closed = False

    while not should_be_closed:
        ret, image = cap.read()
        if capturing_device == 'cam':
            image = cv2.flip(image, 1)

        M = OF.evaluate_flow(image)
        cv2.imshow(window_name, OF.draw_current_frame())
        OF.next_step()

        key = cv2.waitKey(1)

        if key & 0xFF == 27:
            should_be_closed = True

    if capturing_device == 'cam':
        cap.release()
    cv2.destroyAllWindows()

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_batch():
    filenames = tools_IO.get_filenames(folder_in, '*.jpg')[::-1]
    image = cv2.imread(folder_in + filenames[0])
    faces = numpy.array([None] * len(filenames))
    faces[0] = numpy.array([[1560, 262], [1811, 234], [1768, 736], [1532, 665]])
    faces[5] = numpy.array([[1320, 246], [1683, 214], [1622, 816], [1291, 709]])
    faces[6] = numpy.array([[1257, 228], [1647, 196], [1600, 825], [1231, 714]])
    faces[15] = numpy.array([[878, 147], [1468, 124], [1404, 925], [864, 793]])

    OF = tools_optical_flow.OpticalFlow_LucasKanade(image, face_2d=faces[0], folder_out=folder_out)

    for filename, face in zip(filenames, faces):
        image = cv2.imread(folder_in + filename)

        OF.update_ROI(face)
        M = OF.evaluate_flow(image)
        if face is not None:
            OF.face_2d_cur = face

        cv2.imwrite(folder_out + filename, OF.draw_current_frame())
        OF.next_step()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out,'*.jpg')
    example_batch()

