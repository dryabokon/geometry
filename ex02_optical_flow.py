import numpy
import cv2
# --------------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
from CV import tools_Skeletone
# --------------------------------------------------------------------------------------------------------------------------
class OpticalFlow_DenseByLines():
    def __init__(self):
        self.step = 26

    def set_baseline(self, frame):
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def get_flow_image(self, frame):
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        h, w = next.shape[:2]

        y, x = numpy.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step].reshape(2, -1)
        fx, fy = flow[y, x].T
        lines = numpy.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = numpy.int32(lines + 0.5)
        result = cv2.cvtColor(next, cv2.COLOR_GRAY2BGR)
        cv2.polylines(result, lines, 0, (0, 0, 200))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(result, (x1, y1), 1, (0, 0, 200), -1)


        # th = 4
        # y, x = np.mgrid[0:h:1, 0:w:1].reshape(2, -1)
        # fx, fy = flow[y, x].T
        # fx = fx.reshape((h,w))
        # fy = fy.reshape((h, w))
        # mask  = 255*(( (fx>th) + (fy>th) ) > 0 )
        # #result = mask
        # result = tools_image.put_color_by_mask(frame,mask,(0,0,200))


        self.prev = next
        return result
# --------------------------------------------------------------------------------------------------------------------------
class OpticalFlow_LucasKanade():
    def __init__(self):

        self.feature_params = dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = numpy.random.randint(0, 255, (100, 3))

    def set_baseline(self, frame):
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.mask = numpy.zeros_like(frame)

    def get_flow_image(self, frame):
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

            img = cv2.add(frame, self.mask)

            # Now update the previous frame and previous points
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
def example_folder(OF,folder_in,folder_out,ID_start = 0):
    tools_IO.remove_files(folder_out)

    filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    baseline = cv2.imread(folder_in + filenames[ID_start])
    #baseline = tools_image.saturate(S.binarize(baseline))
    OF.set_baseline(baseline)

    for filename in filenames[ID_start:]:
        image = cv2.imread(folder_in + filename)
        #image = tools_image.saturate(S.binarize(image))
        result = OF.get_flow_image(image)
        cv2.imwrite(folder_out+filename,result)

    return
# --------------------------------------------------------------------------------------------------------------------------
OF_lines = OpticalFlow_DenseByLines()
OF_LK = OpticalFlow_LucasKanade()
# --------------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
folder_in = './images/ex_optical_flow/'
S = tools_Skeletone.Skelenonizer(folder_out)
# --------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #example_cam(OF_LK)
    example_folder(OF_lines,folder_in,folder_out)