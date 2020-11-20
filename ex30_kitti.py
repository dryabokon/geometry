from os import listdir
import fnmatch
import cv2
import os
import itertools
from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
import numpy
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
import matplotlib.patches as patches
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
import tools_GL3D
import tools_pr_geom
import tools_render_CV
import tools_IO
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
class Tracklet(object):
    """
    Representation an annotated object track

    Tracklets are created in function parseXML and can most conveniently used as follows:

    for trackletObj in parseXML(trackletFile):
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
      ... your code here ...
    #end: for all frames
    #end: for all tracklets

    absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
    amtOcclusion and amtBorders could be None

    You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int),
    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
    """

    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None   # n x 3 float array (x,y,z)
    rots = None    # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        """
        Creates Tracklet with no info set
        """
        self.size = numpy.nan * numpy.ones(3, dtype=float)

    def __str__(self):
        """
        Returns human-readable string representation of tracklet object

        called implicitly in
        print trackletObj
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

    def __iter__(self):
        """
        Returns an iterator that yields tuple of all the available data for each frame

        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amtOccs is None:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None), range(self.firstFrame, self.firstFrame+self.nFrames))
        else:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame+self.nFrames))
# ----------------------------------------------------------------------------------------------------------------------
def parseXML(trackletFile):
    """
    Parses tracklet xml file and convert results to list of Tracklet objects

    :param trackletFile: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """

    # convert tracklet XML data to a tree structure

    STATE_UNSET = 0
    STATE_INTERP = 1
    STATE_LABELED = 2
    OCC_UNSET = 255  # -1 as uint8
    OCC_VISIBLE = 0
    OCC_PARTLY = 1
    OCC_FULLY = 2
    TRUNC_IN_IMAGE = 0
    TRUNC_TRUNCATED = 1
    TRUNC_OUT_IMAGE = 2
    TRUNC_BEHIND_IMAGE = 3
    stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}
    occFromText = {'-1': OCC_UNSET, '0': OCC_VISIBLE, '1': OCC_PARTLY, '2': OCC_FULLY}
    TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
    truncFromText = {'99': TRUNC_UNSET, '0': TRUNC_IN_IMAGE, '1': TRUNC_TRUNCATED, '2': TRUNC_OUT_IMAGE,'3': TRUNC_BEHIND_IMAGE}

    eTree = ElementTree()
    print('Parsing tracklet file', trackletFile)
    with open(trackletFile) as f:
        eTree.parse(f)

    # now convert output to list of Tracklet objects
    trackletsElem = eTree.find('tracklets')
    tracklets = []
    trackletIdx = 0
    nTracklets = None
    for trackletElem in trackletsElem:
        #print 'track:', trackletElem.tag
        if trackletElem.tag == 'count':
            nTracklets = int(trackletElem.text)
            print('File contains', nTracklets, 'tracklets')
        elif trackletElem.tag == 'item_version':
            pass
        elif trackletElem.tag == 'item':
            #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
            # a tracklet
            newTrack = Tracklet()
            isFinished = False
            hasAmt = False
            frameIdx = None
            for info in trackletElem:
                #print 'trackInfo:', info.tag
                if isFinished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    newTrack.objectType = info.text
                elif info.tag == 'h':
                    newTrack.size[0] = float(info.text)
                elif info.tag == 'w':
                    newTrack.size[1] = float(info.text)
                elif info.tag == 'l':
                    newTrack.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    newTrack.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        #print 'trackInfoPose:', pose.tag
                        if pose.tag == 'count':     # this should come before the others
                            if newTrack.nFrames is not None:
                                raise ValueError('there are several pose lists for a single track!')
                            elif frameIdx is not None:
                                raise ValueError('?!')
                            newTrack.nFrames = int(pose.text)
                            newTrack.trans = numpy.nan * numpy.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.rots = numpy.nan * numpy.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.states = numpy.nan * numpy.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.occs = numpy.nan * numpy.ones((newTrack.nFrames, 2), dtype='uint8')
                            newTrack.truncs = numpy.nan * numpy.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.amtOccs = numpy.nan * numpy.ones((newTrack.nFrames, 2), dtype=float)
                            newTrack.amtBorders = numpy.nan * numpy.ones((newTrack.nFrames, 3), dtype=float)
                            frameIdx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frameIdx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                #print 'trackInfoPoseInfo:', poseInfo.tag
                                if poseInfo.tag == 'tx':
                                    newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                                    hasAmt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frameIdx += 1
                        else:
                            raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    isFinished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
            #end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not isFinished:
                warn('tracklet {0} was not finished!'.format(trackletIdx))
            if newTrack.nFrames is None:
                warn('tracklet {0} contains no information!'.format(trackletIdx))
            elif frameIdx != newTrack.nFrames:
                warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(trackletIdx, newTrack.nFrames, frameIdx))
            if numpy.abs(newTrack.rots[:, :2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amtOccs / amtBorders are not set, set them to None
            if not hasAmt:
                newTrack.amtOccs = None
                newTrack.amtBorders = None

            # add new tracklet to list
            tracklets.append(newTrack)
            trackletIdx += 1

        else:
            raise ValueError('unexpected tracklet info')
    #end: for tracklet list items

    print('Loaded', trackletIdx, 'tracklets.')

    # final consistency check
    if trackletIdx != nTracklets:
        warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets, trackletIdx))

    return tracklets
# ----------------------------------------------------------------------------------------------------------------------
def local_ori(trans, rot):
    #compute local orientation value based on global orientation and translation values
    local_ori = rot - numpy.arctan(trans[0] / trans[2])
    return round(local_ori,2)
# ----------------------------------------------------------------------------------------------------------------------
def obtain_2Dbox(dims, trans, rot, P2, img_xmax, img_ymax):
    '''
    obtain 2D bounding box based on 3D location values
    construct 3D bounding box at first, 2D bounding box is just the minimal and maximal values of 3D bounding box
    '''
    # generate 8 points for bounding box
    h, w, l = dims[0], dims[1], dims[2]
    tx, ty, tz = trans[0], trans[1], trans[2]

    R = numpy.array([[numpy.cos(rot), 0, numpy.sin(rot)],
                     [0, 1, 0],
                     [-numpy.sin(rot), 0, numpy.cos(rot)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = numpy.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += numpy.array([tx, ty, tz]).reshape((3, 1))

    corners_3D_1 = numpy.vstack((corners_3D, numpy.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    for i in range(len(corners_2D[0, :])):
        if corners_2D[0, i] < 0:
            corners_2D[0, i] = 0
        elif corners_2D[0, i] > img_xmax:
            corners_2D[0, i] = img_xmax

    for j in range(len(corners_2D[1, :])):
        if corners_2D[1, j] < 0:
            corners_2D[1, j] = 0
        elif corners_2D[1, j] > img_ymax:
            corners_2D[1, j] = img_ymax

    xmin, xmax = int(min(corners_2D[0, :])), int(max(corners_2D[0, :]))
    ymin, ymax = int(min(corners_2D[1, :])), int(max(corners_2D[1, :]))

    bbox = [xmin, ymin, xmax, ymax]

    return bbox
# ----------------------------------------------------------------------------------------------------------------------
def get_labels_from_tracklets(mat_transform, P2, folder_in, folder_out, folder_images):

    if not os.path.exists(folder_out): os.mkdir(folder_out)

    for trackletObj in parseXML(os.path.join(folder_in, 'tracklet_labels.xml')):
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
            label_file = folder_out + str(absoluteFrameNumber).zfill(10) + '.txt'
            image_file = folder_images + str(absoluteFrameNumber).zfill(10) + '.png'
            img = cv2.imread(image_file)
            img_xmax, img_ymax = img.shape[1], img.shape[0]

            translation = numpy.append(translation, 1)
            translation = numpy.dot(mat_transform, translation)
            translation = translation[:3]/translation[3]

            rot = -(rotation[2] + numpy.pi / 2)
            if rot > numpy.pi:
                rot -= 2 * numpy.pi
            elif rot < -numpy.pi:
                rot += 2 * numpy.pi
            rot = round(rot, 2)

            local_rot = local_ori(translation, rot)


            bbox = obtain_2Dbox(trackletObj.size, translation, rot, P2, img_xmax, img_ymax)

            with open(label_file, 'a') as file_writer:
                line = [trackletObj.objectType] + [int(truncation),int(occlusion[0]),local_rot] + bbox + [round(size, 2) for size in trackletObj.size] \
                + [round(tran, 2) for tran in translation] + [rot]
                line = ' '.join([str(item) for item in line]) + '\n'
                file_writer.write(line)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_calib_from_tracklets(line_P2, folder_out):
    if not os.path.exists(folder_out): os.mkdir(folder_out)
    for image in os.listdir(folder_images):
        calib_file = folder_out + image.split('.')[0] + '.txt'

        with open(calib_file, 'w') as file_writer:
            file_writer.write(line_P2)
    return
# ----------------------------------------------------------------------------------------------------------------------
def read_transformation_matrix(tracklet_path):
    for line in open(os.path.join(tracklet_path, 'calib_velo_to_cam.txt')):
        if 'R:' in line:
            R = line.strip().split(' ')
            R = numpy.asarray([float(number) for number in R[1:]])
            R = numpy.reshape(R, (3, 3))

        if 'T:' in line:
            T = line.strip().split(' ')
            T = numpy.asarray([float(number) for number in T[1:]])
            T = numpy.reshape(T, (3, 1))

    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'R_rect_00:' in line:
            R0_rect = line.strip().split(' ')
            R0_rect = numpy.asarray([float(number) for number in R0_rect[1:]])
            R0_rect = numpy.reshape(R0_rect, (3, 3))

    # recifying rotation matrix
    R0_rect = numpy.append(R0_rect, numpy.zeros((3, 1)), axis=1)
    R0_rect = numpy.append(R0_rect, numpy.zeros((1, 4)), axis=0)
    R0_rect[-1,-1] = 1

    #The rigid body transformation from Velodyne coordinates to camera coordinates
    Tr_velo_to_cam = numpy.concatenate([R, T], axis=1)
    Tr_velo_to_cam = numpy.append(Tr_velo_to_cam, numpy.zeros((1, 4)), axis=0)
    Tr_velo_to_cam[-1,-1] = 1

    transform = numpy.dot(R0_rect, Tr_velo_to_cam)

    # FIGURE OUT THE CALIBRATION
    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'P_rect_02' in line:
            line_P2 = line.replace('P_rect_02', 'P2')
            # print (line_P2)

    P2 = line_P2.split(' ')
    P2 = numpy.asarray([float(i) for i in P2[1:]])
    P2 = numpy.reshape(P2, (3, 4))

    return transform, line_P2, P2
# ----------------------------------------------------------------------------------------------------------------------
def extract_labels_from_tracklets(folder_tracklets,folder_labels, folder_images):
    mat_transform, line_P2, P2 = read_transformation_matrix(folder_tracklets)
    get_labels_from_tracklets(mat_transform, P2, folder_tracklets, folder_labels, folder_images)
    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_calibs_from_tracklets(folder_tracklets,folder_calib):
    mat_transform, line_P2, P2 = read_transformation_matrix(folder_tracklets)
    get_calib_from_tracklets(line_P2, folder_calib)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_filenames(path_input,list_of_masks):
    local_filenames = []
    for mask in list_of_masks.split(','):
        local_filenames += fnmatch.filter(listdir(path_input), mask)

    return local_filenames
# ----------------------------------------------------------------------------------------------------------------------
class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = numpy.transpose(numpy.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < numpy.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if numpy.pi / 2 <= rot_local <= numpy.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if numpy.pi < rot_local <= 3 / 2 * numpy.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * numpy.pi / 2 <= rot_local <= 2 * numpy.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * numpy.pi / 180
        if 0 < rot_local < div or 2*numpy.pi-div < rot_local < 2*numpy.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if numpy.pi - div < rot_local < numpy.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi
# ----------------------------------------------------------------------------------------------------------------------
def get_cube_3D(record):
    obj = detectionInfo(record)
    R = numpy.array([[numpy.cos(obj.rot_global), 0, numpy.sin(obj.rot_global)], [0, 1, 0], [-numpy.sin(obj.rot_global), 0, numpy.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = numpy.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += numpy.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))
    #corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    return corners_3D.T
# ----------------------------------------------------------------------------------------------------------------------
def project_2D(P2, corners_3D):

    corners_3D_1 = numpy.vstack((corners_3D.T, numpy.ones(((corners_3D.T).shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D.T
# ----------------------------------------------------------------------------------------------------------------------
def project_2D_BEV_old(line,scale,shape):

    npline = [numpy.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = numpy.array([[-numpy.cos(rot_y), numpy.sin(rot_y)], [numpy.sin(rot_y), numpy.cos(rot_y)]])
    t = numpy.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -l / 2

    corners_2D = numpy.array([x_corners, z_corners])
    corners_2D = R.dot(corners_2D)
    corners_2D = t - corners_2D
    corners_2D[0] += int(shape / 2)
    corners_2D = (corners_2D).astype(numpy.int16)

    return corners_2D.T
# ----------------------------------------------------------------------------------------------------------------------
def draw_3Dbox(ax, P2, line, color):

    corners_2D = project_2D(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)
# ----------------------------------------------------------------------------------------------------------------------
def points_to_lines(points):

    idx = [(0,1),(1,2),(2,7),(7,0),(3,4),(4,5),(5,6),(6,3),(0,1),(1,4),(4,5),(5,0),(2,3),(3,6),(6,7),(7,2),(0,5),(5,6),(6,7),(7,0),(1,2),(2,3),(3,4),(4,1)]
    lines = []
    for i in idx:
        lines.append(numpy.array((points[i[0]],points[i[1]])).flatten())

    return lines
# ----------------------------------------------------------------------------------------------------------------------
def get_h_ipersp(image,target_W,target_H,point_van_xy = (631, 166),):
    H, W = image.shape[:2]

    tol_up = 20
    tol_bottom = 0
    upper_line = (0, point_van_xy[1] + tol_up, W, point_van_xy[1] + tol_up)
    bottom_line = (0, H - tol_bottom, W, H - tol_bottom)

    pad_left = 1 * W
    pad_right = 1 * W
    line1 = (-pad_left, H, point_van_xy[0], point_van_xy[1])
    line2 = (W + pad_right, H, point_van_xy[0], point_van_xy[1])
    p1 = tools_render_CV.line_intersection(upper_line, line1)
    p2 = tools_render_CV.line_intersection(upper_line, line2)
    p3 = tools_render_CV.line_intersection(bottom_line, line1)
    p4 = tools_render_CV.line_intersection(bottom_line, line2)
    src = numpy.array([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])], dtype=numpy.float32)
    dst = numpy.array([(0, 0), (target_W, 0), (0, target_H), (target_W, target_H)], dtype=numpy.float32)
    #image = tools_draw_numpy.draw_convex_hull(image, [p1, p2, p3, p4], color=(36, 10, 48), transperency=0.5)
    h_ipersp = cv2.getPerspectiveTransform(src, dst)

    return h_ipersp
# ----------------------------------------------------------------------------------------------------------------------
def project_2D_BEV(X_source,H):
    res = cv2.perspectiveTransform(X_source.reshape(-1, 1, 2), H).reshape((-1, 2))
    return res
# ----------------------------------------------------------------------------------------------------------------------
def draw_boxes(folder_out, folder_images, folder_labels_GT, folder_calib):

    tools_IO.remove_files(folder_out,create=True)
    local_filenames = get_filenames(folder_images, '*.png,*.jpg')

    for index,local_filename in enumerate(local_filenames):
        base_name = local_filename.split('/')[-1].split('.')[0]
        print(base_name)
        filename_image = folder_images + local_filename
        filename_label = folder_labels_GT + base_name + '.txt'
        filename_calib = folder_calib + base_name + '.txt'

        record = open(filename_calib).readlines()[0]
        mat_proj = numpy.asarray([float(i) for i in record.split(' ')[1:]]).reshape((3, 4))

        image = tools_image.desaturate(cv2.imread(filename_image))
        H,W = image.shape[:2]
        target_BEV_W, target_BEV_H = int(H*0.75),H


        records = open(filename_label).readlines()
        colors = tools_draw_numpy.get_colors(len(records))
        image_2d = image.copy()
        h_ipersp = get_h_ipersp(image,target_BEV_W, target_BEV_H)
        image_BEV = cv2.warpPerspective(image, h_ipersp, (target_BEV_W, target_BEV_H), borderValue=(32, 32, 32))
        for record,color in zip(records,colors):
            record = record.strip().split(' ')

            points_3d = get_cube_3D(record)
            points_2d = project_2D(mat_proj, points_3d)
            points_2d_BEV = project_2D_BEV(points_2d[[2,3,6,7]],h_ipersp)
            lines_2d = points_to_lines(points_2d)

            #image = tools_draw_numpy.draw_lines(image,lines_2d,color=color.tolist(),w=1)
            #image = tools_render_CV.draw_points_numpy_RT(points_3d, image,mat_proj,camera_matrix_4x4, flipX=True)
            #image = tools_render_CV.draw_points_numpy_RT(points_3d, image,PP      ,numpy.eye(3)     , flipX=True)
            #image = R.get_image_perspective_M(PP,aperture,aperture,scale=1*numpy.array((1,1,1)),lookback=False,do_debug=True)

            #image = tools_render_CV.draw_cube_numpy_RT(points_3d,image,PP,numpy.eye(3),color=color,flipX=True)
            image_2d = tools_draw_numpy.draw_convex_hull_cv(image_2d,points_2d,color.tolist(),transperency=0.75)
            image_2d = tools_draw_numpy.draw_lines(image_2d,lines_2d,color.tolist(),w=1)
            image_BEV = tools_draw_numpy.draw_convex_hull_cv(image_BEV, points_2d_BEV, color.tolist(), transperency=0.25)


        image_result = numpy.zeros((H,W+target_BEV_W,3),dtype=numpy.uint8)
        image_result[:,:W]=image_2d
        image_result[:,W:]=image_BEV

        cv2.imwrite(folder_out + base_name + '.png', image_result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_boxes_from_labels(folder_out, folder_images, folder_labels_GT):
    if not os.path.exists(folder_out): os.mkdir(folder_out)
    local_filenames = get_filenames(folder_images, '*.png,*.jpg')

    for index,local_filename in enumerate(local_filenames):
        base_name = local_filename.split('/')[-1].split('.')[0]
        filename_label = folder_labels_GT + base_name + '.txt'

        cubes = []
        with open(filename_label) as f1:
            for line_p in f1:
                line_p = line_p.strip().split(' ')
                cube = get_cube_3D(line_p)
                cubes.append(cube.flatten())

        tools_IO.save_mat(cubes,folder_out+base_name+'_cube.txt')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_tracklets = '../kitti_dataset/2011_09_26/2011_09_26_drive_0001_sync/'
    folder_labels = folder_tracklets + 'label_02/'
    folder_images = folder_tracklets + 'image_02/data/'
    folder_calib  = folder_tracklets + 'calib_02/'
    folder_out = folder_tracklets + 'output/'

    #extract_labels_from_tracklets(folder_tracklets,folder_labels,folder_images)
    #extract_calibs_from_tracklets(folder_tracklets,folder_calib)
    #extract_boxes_from_labels(folder_out, folder_images, folder_labels)
    #draw_boxes(folder_out, folder_images, folder_labels, folder_calib)
    #tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'kitti_00.gif', mask='*.png', framerate=10,resize_H=int(375*0.75), resize_W=int(1429*0.75),do_reverce=False)
    tools_animation.folder_to_video(folder_out,folder_out+'kitti_00.mp4',mask='*.png',framerate=10)