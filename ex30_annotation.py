# ---------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
import os
from os import listdir
import fnmatch
import tools_IO
import tools_render_CV
import tools_image
import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------------------------------------------------
class markup:

    def __init__(self, folder,file_markup,file_classes):

        self.scale = 0.70
        self.draw_labels = False

        if os.path.isfile(file_classes):
            self.class_names = tools_IO.load_mat(folder + file_classes, numpy.str)
        else:
            self.class_names = ['class %02d'%i for i in range(16)]

        self.colors = self.get_colors(len(self.class_names))

        self.input_folder = folder
        self.output_file = folder+file_markup
        self.textcolor = (0,0,0)

        self.filenames = fnmatch.filter(listdir(self.input_folder), '*.jpg')
        self.filenames.sort()
        self.num_filenames = len(self.filenames)

        self.current_frame_ID=0
        self.image_to_display = None
        self.current_markup_ID = 0

        self.frameID = []
        self.lines = []
        self.class_ID = []

        self.load_markup_lines()
        self.last_insert_size=1

        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def save_markup_lines(self):
        res = [('filename', 'x1', 'y1', 'x2', 'y2', 'class_ID')]

        for ID, filename in enumerate(self.filenames):
            for i in range(0, len(self.frameID)):
                if self.frameID[i] == ID:
                    res.append([self.filenames[self.frameID[i]], self.lines[i][0], self.lines[i][1], self.lines[i][2], self.lines[i][3], self.class_ID[i]])
        tools_IO.save_mat(res, self.output_file, delim=' ')
        return
# ---------------------------------------------------------------------------------------------------------------------
    def load_markup_lines(self):
        if not os.path.isfile(self.output_file):return
        if tools_IO.count_lines(self.output_file)<=1:return
        for each in tools_IO.load_mat(self.output_file,delim=' ')[1:]:
            i= tools_IO.smart_index(self.filenames, each[0].decode("utf-8"))
            if len(i)>0:
                i=i[0]
                self.frameID.append(i)
                self.lines.append([int(each[1]),int(each[2]),int(each[3]),int(each[4])])
                self.class_ID.append(int(each[5]))
        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_colors(self,N,alpha_blend=None):
        colors = []
        for i in range(0, N):
            hue = int(255 * i / (N-1))
            color = cv2.cvtColor(numpy.array([hue, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0][0]
            if alpha_blend is not None:color = ((alpha_blend) * numpy.array((255, 255, 255)) + (1-alpha_blend) * numpy.array(color))
            colors.append((int(color[0]), int(color[1]), int(color[2])))

        return colors
# ---------------------------------------------------------------------------------------------------------------------
    def refresh_image(self):
        self.image_to_display = cv2.imread(self.input_folder + self.filenames[self.current_frame_ID])
        self.image_to_display = tools_image.desaturate(self.image_to_display, level=0.9)

        self.image_to_display = self.draw_lines(self.image_to_display,draw_labels=self.draw_labels)
        self.image_to_display = self.draw_legend(self.image_to_display)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def goto_next_frame(self):
        self.current_frame_ID+=1
        if self.current_frame_ID>=len(self.filenames):self.current_frame_ID = 0
        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def goto_prev_frame(self):
        self.current_frame_ID-=1
        if self.current_frame_ID<0:self.current_frame_ID = len(self.filenames) - 1
        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_next_marked(self,same_class=False):

        start, success = self.current_frame_ID, False
        current = self.current_frame_ID

        while not success:
            current += 1
            if current>=len(self.filenames):current = 0
            if current in self.frameID:
                if not same_class:
                    success = True
                else:
                    idx = numpy.where(numpy.array(self.frameID) == current)
                    labels = numpy.array(self.class_ID)[idx]
                    if self.current_markup_ID in labels:
                        success = True
            if current == start:
                success = True

        return current
# ---------------------------------------------------------------------------------------------------------------------
    def get_prev_marked(self,same_class=False):

        start, success = self.current_frame_ID, False
        current = self.current_frame_ID

        while not success:
            current -= 1
            if current < 0: current = len(self.filenames) - 1
            if current in self.frameID:
                if not same_class:
                    success = True
                else:
                    idx = numpy.where(numpy.array(self.frameID)==current)
                    labels = numpy.array(self.class_ID)[idx]
                    if self.current_markup_ID in labels:
                        success = True

            if current == start:
                success = True

        return current
# ---------------------------------------------------------------------------------------------------------------------
    def get_objects(self,frame_ID,classID=None):

        idx = numpy.where(numpy.array(self.frameID) == frame_ID)
        classIDs = numpy.array(self.class_ID)[idx]
        lines    = numpy.array(self.lines)[idx]

        if classID is None:
            return lines.tolist(), classIDs.tolist()
        else:
            classIDs = numpy.array(classIDs)
            lines = numpy.array(lines)
            idx = numpy.where(classIDs==classID)
            return lines[idx].tolist(), classIDs[idx].tolist()
# ---------------------------------------------------------------------------------------------------------------------
    def add_lines(self):

        next_frame_ID = self.get_next_marked(same_class=True)
        if next_frame_ID <= self.current_frame_ID:return

        lines_current, classIDs = self.get_objects(self.current_frame_ID,self.current_markup_ID)
        lines_next, classIDs = self.get_objects(next_frame_ID, self.current_markup_ID)

        if not (len(lines_current)==1 and len(lines_next)==1):
            return


        for frameID in range(self.current_frame_ID+1,next_frame_ID):
            alpha =  float((next_frame_ID-1-frameID)/(next_frame_ID-1-(self.current_frame_ID+1)))
            line = numpy.array(lines_current[0]).astype(float)*(alpha) + numpy.array(lines_next[0]).astype(float)*(1-alpha)
            M.lines.append(line.astype(int).tolist())
            M.frameID.append(frameID)
            M.class_ID.append(self.current_markup_ID)

        self.last_insert_size = next_frame_ID-self.current_frame_ID-1
        self.current_frame_ID = next_frame_ID

        return
# ---------------------------------------------------------------------------------------------------------------------
    def draw_lines(self,image,draw_labels=True):
        for i, each in enumerate(self.frameID):
            if each == self.current_frame_ID:
                id = self.class_ID[i]
                left,top,right,bottom = self.lines[i][0], self.lines[i][1], self.lines[i][2], self.lines[i][3]
                cv2.line(image, (left, top), (right, bottom), self.colors[id], 2)
                if draw_labels:
                    cv2.putText(image, '{0:d}'.format(id), ((left+right)//2,(top+bottom)//2),cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[id], 1, cv2.LINE_AA)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_legend(self, image):
        if self.textcolor==(0,0,0):
            thikness = 1
        else:
            thikness  = 2
        color =  self.colors[self.current_markup_ID]
        class_name = self.class_names[self.current_markup_ID]
        cv2.rectangle(image,(0,0),(200,50),(255,255,255),-1)
        cv2.putText(image, '{0:d} {1:s}'.format(self.current_markup_ID,class_name), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(image, '{0:d} {1:s}'.format(self.current_frame_ID,self.filenames[self.current_frame_ID]), (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textcolor, thikness, cv2.LINE_AA)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def add_line(self):
        top    = int(g_coord[0][1]/self.scale)
        bottom = int(g_coord[1][1]/self.scale)
        left   = int(g_coord[0][0]/self.scale)
        right  = int(g_coord[1][0]/self.scale)

        M.lines.append([left,top,right,bottom])
        M.frameID.append(M.current_frame_ID)
        M.class_ID.append(M.current_markup_ID)
        M.last_insert_size = 1
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_line(self):

        top    = g_coord[0][1]/self.scale
        left   = g_coord[0][0]/self.scale
        bottom = g_coord[1][1]/self.scale
        right  = g_coord[1][0]/self.scale

        cut = (left,top,right,bottom)

        for i, each in enumerate(self.frameID):
            if each == self.current_frame_ID:
                ln = (M.lines[i][0], M.lines[i][1], M.lines[i][2], M.lines[i][3])
                if tools_render_CV.do_lines_intersect(ln,cut):
                    del M.lines[i]
                    del M.frameID[i]
                    del M.class_ID[i]
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_last_object(self,N=1):
        if N==1 and len(M.lines)>=1:
            del M.lines[-1]
            del M.frameID[-1]
            del M.class_ID[-1]
            return

        if N>1 and len(self.lines)>=N:
            del M.lines[-1]
            del M.frameID[-1]
            del M.class_ID[-1]
            self.remove_last_object(N-1)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_all_from_current_frame(self):

        success = False
        while not success:
            success = True
            idx = tools_IO.smart_index(self.frameID, self.current_frame_ID)
            if len(idx)>0:
                del M.lines[idx[0]]
                del M.frameID[idx[0]]
                del M.class_ID[idx[0]]
                success = False

        return
# ---------------------------------------------------------------------------------------------------------------------
# =====================================================================================================================
g_coord, g_mouse_event_code = [], None
# ---------------------------------------------------------------------------------------------------------------------
def click_handler(event, x, y, flags, param):

    global g_coord, g_mouse_event_code

    is_ctrl  = (flags&0x0F)>0
    is_shift = (flags&0x10)>0
    is_alt   = (flags&0x20)>0


    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None:
        g_coord.append((x, y))
        g_mouse_event_code = 'LBUTTONDOWN'

    elif event == cv2.EVENT_LBUTTONUP and g_mouse_event_code=='LBUTTONDOWN':
        g_coord.append((x, y))
        g_mouse_event_code = 'LBUTTONUP'

    elif event == cv2.EVENT_RBUTTONDBLCLK and g_mouse_event_code is None:
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONDBLCLK'

    elif event == cv2.EVENT_RBUTTONDOWN and g_mouse_event_code is None:
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONDOWN'

    elif event == cv2.EVENT_RBUTTONUP and g_mouse_event_code=='RBUTTONDOWN':
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONUP'

    if event == cv2.EVENT_MOUSEWHEEL:
        if is_ctrl:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_CTL_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_CTL_BK'
        elif is_shift:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_SHF_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_SHF_BK'
        elif is_alt:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_ALT_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_ALT_BK'
        else:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_BK'

    return
# ---------------------------------------------------------------------------------------------------------------------
def application_loop_lines():
    global g_coord, g_mouse_event_code
    while (True):
        if g_mouse_event_code=='LBUTTONUP':
            #TODO - add line
            M.add_line()

            g_mouse_event_code = None
            g_coord.pop()
            g_coord.pop()
            M.refresh_image()
            M.save_markup_lines()
        if g_mouse_event_code == 'RBUTTONUP':
            # TODO - remove line
            M.remove_line()
            g_coord.pop()
            g_coord.pop()
            g_mouse_event_code = None
            M.refresh_image()
            M.save_markup_lines()

        if g_mouse_event_code == 'MOUSEWHEEL_FW':
            g_mouse_event_code = None
            M.goto_prev_frame()
        if g_mouse_event_code == 'MOUSEWHEEL_BK':
            M.goto_next_frame()
            g_mouse_event_code = None


        if g_mouse_event_code == 'MOUSEWHEEL_CTL_FW':
            g_mouse_event_code = None
            M.current_frame_ID = M.get_prev_marked()
            M.refresh_image()

        if g_mouse_event_code == 'MOUSEWHEEL_CTL_BK':
            g_mouse_event_code = None
            M.current_frame_ID = M.get_next_marked()
            M.refresh_image()


        if g_mouse_event_code == 'MOUSEWHEEL_SHF_FW':
            g_mouse_event_code = None
            M.current_frame_ID = M.get_prev_marked(same_class=True)
            M.refresh_image()

        if g_mouse_event_code == 'MOUSEWHEEL_SHF_BK':
            g_mouse_event_code = None
            M.current_frame_ID = M.get_next_marked(same_class=True)
            M.refresh_image()

        if g_mouse_event_code == 'MOUSEWHEEL_ALT_BK':
            g_mouse_event_code = None
            M.add_lines()
            M.save_markup_lines()
            M.refresh_image()


        resized = cv2.resize(M.image_to_display,(int(M.scale*M.image_to_display.shape[1]),int(M.scale*M.image_to_display.shape[0])))
        cv2.imshow(window_name, resized)

        key = cv2.waitKey(1)

        if key & 0xFF == 27:break
        if key & 0xFF == ord('w'):
            M.current_markup_ID = (M.current_markup_ID + 1)%len(M.colors)
            M.refresh_image()
        if key & 0xFF == ord('q'):
            M.current_markup_ID = (M.current_markup_ID - 1)%len(M.colors)
            M.refresh_image()

        if key & 0xFF == ord('l'):
            M.draw_labels = not M.draw_labels
            M.refresh_image()

        if key & 0xFF >= ord('0') and key & 0xFF <= ord('9'):
            M.current_markup_ID = (key & 0xFF)-ord('0')
            M.refresh_image()

        if key == 26:
            M.remove_last_object(M.last_insert_size)
            M.refresh_image()
            M.save_markup_lines()

        if key == 0:
            M.remove_all_from_current_frame()
            M.refresh_image()
            M.save_markup_lines()

    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
folder = './images/ex_lines3/'
# ---------------------------------------------------------------------------------------------------------------------
file_markup ='annotation.txt'
file_classes = 'classes.txt'
M = markup(folder,file_markup,file_classes)
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    window_name = 'image_markup'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_handler)
    application_loop_lines()

