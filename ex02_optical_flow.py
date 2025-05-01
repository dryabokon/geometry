import numpy
import cv2
from tqdm import tqdm
# --------------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
# --------------------------------------------------------------------------------------------------------------------------
class OpticalFlow_DenseByLines():
    def __init__(self):
        return

    def set_baseline(self, frame):
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def get_flow_image(self, frame):
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        h, w = next.shape[:2]

        # self.step = 16
        # y, x = numpy.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step].reshape(2, -1)
        # fx, fy = flow[y, x].T
        # lines = numpy.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        # lines = numpy.int32(lines + 0.5)
        # result = cv2.cvtColor(next, cv2.COLOR_GRAY2BGR)
        # cv2.polylines(result, lines, 0, (0, 0, 200))
        # for (x1, y1), (x2, y2) in lines:
        #     cv2.circle(result, (x1, y1), 1, (0, 0, 200), -1)


        th = 1
        y, x = numpy.mgrid[0:h:1, 0:w:1].reshape(2, -1)
        fx, fy = numpy.abs(flow[y, x].T)
        fx = fx.reshape((h,w))
        fy = fy.reshape((h, w))
        mask  = 255*(( (fx>th) + (fy>th) ) > 0 )
        mask = tools_image.saturate(mask)
        mask[:,:,2]=0
        result = cv2.addWeighted(tools_image.desaturate(frame), 0.7, mask, 0.3, 0)
        #result = tools_image.put_color_by_mask(frame,mask,(0,0,200))


        self.prev = next
        return result
# --------------------------------------------------------------------------------------------------------------------------
class OpticalFlow_LucasKanade():
    def __init__(self):

        self.maxCorners = 200
        self.feature_params = dict(maxCorners=self.maxCorners,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = numpy.random.randint(0, 255, (self.maxCorners, 3))
        self.cnt = 0

    def set_baseline(self, frame):
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.mask = numpy.zeros_like(frame)

    def get_flow_image(self, frame):
        self.cnt+=1
        if self.cnt%10==0:
            self.set_baseline(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,self.p0, None, **self.lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            frame = tools_image.desaturate(frame, 0.9)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.astype(numpy.int32).ravel()
                c, d = old.astype(numpy.int32).ravel()
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)

                frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)

            #img = cv2.add(frame, self.mask)
            img = frame
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

        else:
            img = frame

        return img
# --------------------------------------------------------------------------------------------------------------------------
def example_cam(OF):

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("window")

    rval, frame = cap.read()
    frame = cv2.flip(frame, 1)

    OF.set_baseline(frame)

    while rval:
        rval, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img = OF.get_flow_image(frame)
        cv2.imshow("window", img)

        key = cv2.waitKey(1)
        if key == 27: break
        if key == ord('r'):
            OF.set_baseline(frame)

    cap.release()
    cv2.destroyWindow("window")
    return
# --------------------------------------------------------------------------------------------------------------------------
def example_folder(OF,source,folder_out,ID_start = 0):
    if ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower()):mode = 'video'
    elif ('https' in source) or (source == '0'):mode = 'stream'
    else:mode = 'folder'

    if mode == 'video':
        vidcap = cv2.VideoCapture(source)
        #scroll to the start frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, ID_start)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))-ID_start
        for i in tqdm(range(total_frames), total=total_frames):
            success, image = vidcap.read()
            image = cv2.resize(image, (1280, 720))
            if i == 0:
                OF.set_baseline(image)

            if not success: continue
            result = OF.get_flow_image(image)
            cv2.imwrite(folder_out + 'frame_%06d.jpg'%(i+ID_start), result)
        vidcap.release()

    elif mode == 'folder':
        filenames = tools_IO.get_filenames(source, '*.jpg,*.png')
        baseline = cv2.imread(source + filenames[ID_start])
        OF.set_baseline(baseline)
        for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
            image = cv2.imread(source+filename)
            result = OF.get_flow_image(image)
            cv2.imwrite(folder_out+filename,result)
    return
# --------------------------------------------------------------------------------------------------------------------------
def example_folder2(folder_in,folder_out,bbox,ID_start=0):
    tools_IO.remove_files(folder_out)

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = 'KCF'
    if tracker_type == 'KCF': tracker = cv2.TrackerKCF.create()
    if tracker_type == "CSRT": tracker = cv2.TrackerCSRT.create()

    # if tracker_type == 'BOOSTING': tracker = cv2.TrackerBoosting.create()
    # if tracker_type == 'MIL': tracker = cv2.TrackerMIL_create()
    # if tracker_type == 'TLD': tracker = cv2.TrackerTLD.create()
    # if tracker_type == 'MEDIANFLOW': tracker = cv2.TrackerMedianFlow.create()
    # if tracker_type == 'MOSSE': tracker = cv2.TrackerMOSSE.create()


    filenames = tools_IO.get_filenames(folder_in, '*.jpg,*.png')[ID_start:]
    baseline = cv2.imread(folder_in + filenames[0])
    tracker.init(baseline, bbox)

    for filename in tqdm(filenames,total=len(filenames)):
        image = cv2.imread(folder_in + filename)
        ok, bbox = tracker.update(image)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        image = tools_draw_numpy.draw_rect_fast(image,p1[0],p1[1],p2[0],p2[1],color=(0, 0, 200),w=1)
        cv2.imwrite(folder_out+filename,image)
    return
# --------------------------------------------------------------------------------------------------------------------------
OF_lines = OpticalFlow_DenseByLines()
OF_LK = OpticalFlow_LucasKanade()
# --------------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#source = './images/ex_contours_cyb/'
#source = './images/ex_optical_flow/Day-Praktik-1_B2.mp4'
#source = './images/ex_optical_flow/Naft11.mp4'
source = './images/ex_optical_flow/TownCentreXVID.mp4'
# --------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out)
    example_folder(OF_lines,source,folder_out,ID_start=320)



