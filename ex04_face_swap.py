import cv2
import numpy
import os
import argparse
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_landmark
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
default_filename_in  = './images/ex_faceswap/01/personB-4.jpg'
default_filename_in2 = './images/ex_faceswap/01/personC-1.jpg'
default_folder_in   = './images/ex_faceswap/02/'
default_folder_out  = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def process_folder(filename_in,folder_in,folder_out):
    tools_landmark.transferface_folder(D,filename_in,folder_in,folder_out)
# ---------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='face swap')

    parser.add_argument('--command',default='process_folder')
    parser.add_argument('--filename_in', default=default_filename_in)
    parser.add_argument('--folder_in'  , default=default_folder_in)
    parser.add_argument('--folder_out' , default=default_folder_out)


    args = parser.parse_args()

    full_folder_out = tools_IO.get_next_folder_out(args.folder_out)
    os.mkdir(full_folder_out)

    if args.command=='process_folder':
        process_folder(args.filename_in,args.folder_in,full_folder_out)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #main()
    res2 = tools_landmark.transferface_first_to_second(D, default_filename_in , default_filename_in2, default_folder_out)
    res1 = tools_landmark.transferface_first_to_second(D, default_filename_in2, default_filename_in , default_folder_out)
    cv2.imwrite(default_folder_out + 'first.jpg' , res2)
    cv2.imwrite(default_folder_out + 'second.jpg', res1)

    #tools_landmark.morph_first_to_second(D,default_filename_in2, default_filename_in,default_folder_out,numpy.arange(0.1,0.9,0.1))


    #tools_landmark.transferface_first_to_second_manual(default_filename_in2, default_filename_in, './images/ex_faceswap/markup_true.txt')
    #tools_landmark.morph_first_to_second_manual(D,default_filename_in2, default_filename_in,'./images/ex_faceswap/markup_true.txt',default_folder_out,numpy.arange(0.1,0.9,0.1))

    #im = cv2.imread('D:\image1.png')
    #im = tools_image.de_vignette(im)
    #im = tools_image.desaturate_2d(im)
    #im = tools_image.hitmap2d_to_viridis(im)

    #scale = 6
    #im = tools_image.fade_left_right(im, 40, im.shape[1]-40)
    #im = tools_image.canvas_extrapolate(im, im.shape[0], im.shape[1] * scale)
    #im = tools_image.fade_header(im,im[0,im.shape[1]//2],20)
    #cv2.imwrite('D:\image1a.png',im)