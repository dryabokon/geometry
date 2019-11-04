import cv2
import numpy
import os
import argparse
from scipy.spatial import Delaunay
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
import tools_landmark
import detector_landmarks
import tools_calibrate
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
default_filename_in  = './images/ex_faceswap/01/personD-1.jpg'
default_filename_in2 = './images/ex_faceswap/01/personF-1.jpg'
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
def demo_live(filename_out):
    image1 = cv2.imread('./images/ex_faceswap/01/personB-3.jpg')
    image2 = cv2.imread('./images/ex_faceswap/01/personC-1.jpg')
    L1_original = D.get_landmarks(image1)
    del_triangles = Delaunay(L1_original).vertices

    cap = cv2.VideoCapture(0)
    i=0

    while (True):
        ret, image2 = cap.read()
        i+=1

        L2_original = D.get_landmarks(image2)
        result2 = tools_landmark.do_transfer(image1, image2, L1_original, L2_original, del_triangles)
        #result2 = D.draw_landmarks(image2)

        cv2.imshow('frame', result2)
        cv2.imwrite('./images/output/image_%03d.jpg'%i,image2)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:break
        if (key & 0xFF == 13) or (key & 0xFF == 32):cv2.imwrite(filename_out, result2)
        if key & 0xFF == ord('1'): image1 = cv2.imread('./images/ex_faceswap/01/personA-2.jpg')
        if key & 0xFF == ord('2'): image1 = cv2.imread('./images/ex_faceswap/01/personB-4.jpg')
        if key & 0xFF == ord('3'): image1 = cv2.imread('./images/ex_faceswap/01/personC-1.jpg')
        if key & 0xFF == ord('4'): image1 = cv2.imread('./images/ex_faceswap/01/personC-2.jpg')
        if key & 0xFF == ord('5'): image1 = cv2.imread('./images/ex_faceswap/01/personD-1.jpg')
        if key & 0xFF == ord('6'): image1 = cv2.imread('./images/ex_faceswap/01/personE-1.jpg')
        if key & 0xFF == ord('7'): image1 = cv2.imread('./images/ex_faceswap/01/personH-2.jpg')
        if key & 0xFF == ord('8'): image1 = cv2.imread('./images/ex_faceswap/01/personL-1.jpg')

        if key & 0xFF == ord('a'): image2 = cv2.imread('./images/ex_faceswap/01/personA-2.jpg')
        if key & 0xFF == ord('b'): image2 = cv2.imread('./images/ex_faceswap/01/personB-4.jpg')
        if key & 0xFF == ord('c'): image2 = cv2.imread('./images/ex_faceswap/01/personC-1.jpg')
        if key & 0xFF == ord('d'): image2 = cv2.imread('./images/ex_faceswap/01/personD-2.jpg')
        if key & 0xFF == ord('e'): image2 = cv2.imread('./images/ex_faceswap/01/personE-1.jpg')
        if key & 0xFF == ord('f'): image2 = cv2.imread('./images/ex_faceswap/01/personF-1.jpg')
        if key & 0xFF == ord('h'): image2 = cv2.imread('./images/ex_faceswap/01/personH-1.jpg')
        if key & 0xFF == ord('j'): image2 = cv2.imread('./images/ex_faceswap/01/personJ-1.jpg')
        if key & 0xFF == ord('k'): image2 = cv2.imread('./images/ex_faceswap/01/personK-1.jpg')
        if key & 0xFF == ord('l'): image2 = cv2.imread('./images/ex_faceswap/01/personL-1.jpg')

        if (key & 0xFF >= ord('1')) and (key & 0xFF <= ord('9')):
            L1_original = D.get_landmarks(image1)
            del_triangles = Delaunay(L1_original).vertices

    cap.release()
    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_auto():
    res2 = tools_landmark.transferface_first_to_second(D, default_filename_in , default_filename_in2, default_folder_out)
    #res1 = tools_landmark.transferface_first_to_second(D, default_filename_in2, default_filename_in , default_folder_out)
    cv2.imwrite(default_folder_out + default_filename_in.split('/')[-1] , res2)
    #cv2.imwrite(default_folder_out + default_filename_in2.split('/')[-1], res1)
    #tools_landmark.morph_first_to_second(D,default_filename_in2, default_filename_in,default_folder_out,numpy.arange(0.1,0.9,0.1))
    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_manual():
    tools_landmark.transferface_first_to_second_manual(default_filename_in, default_filename_in2, './images/ex_faceswap/markup.txt')
    tools_landmark.morph_first_to_second_manual(D,default_filename_in2, default_filename_in,'./images/ex_faceswap/markup.txt',default_folder_out,numpy.arange(0.1,0.9,0.1))
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    demo_live(default_folder_out+'result.jpg')
    #demo_manual()
    #demo_auto()
    #process_folder(default_filename_in,'./images/ex_faceswap/02/',default_folder_out)