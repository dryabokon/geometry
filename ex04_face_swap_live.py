import cv2
# ---------------------------------------------------------------------------------------------------------------------
import time
import tools_IO
from CV import tools_faceswap
from detector import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
camera_W, camera_H = 640, 480
use_camera = True
do_transfer = True
# ---------------------------------------------------------------------------------------------------------------------
def click_handler(event, x, y, flags, param):

    global do_transfer
    global filename_clbrt

    if event == cv2.EVENT_MOUSEWHEEL:
        idx = tools_IO.smart_index(list_filenames, filename_clbrt)[0]
        if flags>0:
            idx = (idx + 1) % len(list_filenames)
        else:
            idx = (idx - 1 + len(list_filenames)) % len(list_filenames)

        filename_clbrt = list_filenames[idx]
        do_transfer = True
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        FS.update_clbrt(image_clbrt)

    return
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):

    global use_camera,do_transfer
    global image_clbrt,image_actor
    global folder_in
    global list_filenames
    global filename_clbrt,filename_actor

    if key == ord('w') or key == ord('s'):
        idx = tools_IO.smart_index(list_filenames, filename_clbrt)[0]
        if key==ord('w'):
            idx  =(idx+1)%len(list_filenames)
        else:
            idx = (idx-1+len(list_filenames)) % len(list_filenames)

        filename_clbrt= list_filenames[idx]
        do_transfer = True
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        FS.update_clbrt(image_clbrt)

    if key==ord('a') or key==ord('d'):
        idx = tools_IO.smart_index(list_filenames, filename_actor)[0]
        if key==ord('d'):
            idx  =(idx+1)%len(list_filenames)
        else:
            idx = (idx-1+len(list_filenames)) % len(list_filenames)
        filename_actor = list_filenames[idx]
        use_camera = False
        image_actor = cv2.imread(folder_in + filename_actor)
        FS.update_actor(image_actor)
        image_clbrt = cv2.imread(folder_in + filename_clbrt)
        FS.update_clbrt(image_clbrt)


    if key==9:
        use_camera = not use_camera

    if key & 0xFF == ord('0') or key & 0xFF == ord('`'):
        if use_camera:
            do_transfer = not do_transfer

    if (key & 0xFF == 13) or (key & 0xFF == 32):
        cv2.imwrite(folder_out+'C.jpg', image_clbrt)
        cv2.imwrite(folder_out+'A.jpg', image_actor)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_live(FS,cam_id=0):

    if use_camera:
        cap = cv2.VideoCapture(cam_id)
        cap.set(3, camera_W)
        cap.set(4, camera_H)
    else:
        cap = None

    #cv2.setMouseCallback('frame', click_handler)

    cnt, start_time, fps = 0, time.time(), 0
    while (True):
        if use_camera:
            if cap is None:
                cap = cv2.VideoCapture(cam_id)
                cap.set(3, camera_W)
                cap.set(4, camera_H)

            ret, image_actor = cap.read()
            image_actor = cv2.flip(image_actor,1)
            FS.update_actor(image_actor)
            if cnt==0:
                FS.update_clbrt(image_clbrt)

        if do_transfer:
            result = FS.do_faceswap()
        else:
            result = image_actor


        if time.time() > start_time: fps = cnt / (time.time() - start_time)

        result = cv2.putText(result, '{0: 1.1f} {1}{2}'.format(fps, ' fps@', FS.device), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1, cv2.LINE_AA)
        result = cv2.putText(result, 'Clbrt: {0}'.format(filename_clbrt.split('/')[-1]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1,cv2.LINE_AA)
        result = cv2.putText(result, 'Actor: {0}'.format(filename_actor.split('/')[-1]), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(128, 128, 0), 1,cv2.LINE_AA)
        cv2.imshow('frame', result)

        cnt += 1
        key = cv2.waitKey(1)
        process_key(key)
        if key & 0xFF == 27: break

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()

    return
# ---------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_faceswap/01/'
folder_out = './images/output1/'
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')[:7]
    filename_clbrt, filename_actor = list_filenames[-1], list_filenames[ 1]
    image_clbrt = cv2.imread(folder_in+filename_clbrt)
    image_actor = cv2.imread(folder_in+filename_actor)

    FS = tools_faceswap.Face_Swapper(D, image_clbrt, image_actor, device='cpu', adjust_every_frame=True, do_narrow_face=True)
    demo_live(FS)

