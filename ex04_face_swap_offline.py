import numpy
import cv2
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
from CV import tools_faceswap
from detector import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ---------------------------------------------------------------------------------------------------------------------
def demo_01(folder_out,filename_clbrt,filename_actor):
    image_clbrt = cv2.imread(filename_clbrt)
    image_actor = cv2.imread(filename_actor)
    FS = tools_faceswap.Face_Swapper(D, image_clbrt, image_actor, device='cpu')
    FS.update_clbrt(image_clbrt)

    result = FS.do_faceswap(folder_out=folder_out, do_debug=False)
    cv2.imwrite(folder_out + 'result.jpg' , result)

    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_02(folder_out,folder_in,list_filenames):

    FS = tools_faceswap.Face_Swapper(D, numpy.zeros((10, 10, 3), dtype=numpy.uint8), numpy.zeros((10, 10, 3), dtype=numpy.uint8), device='cpu')

    for j, filename_actor in enumerate(list_filenames):
        image_actor = cv2.imread(folder_in + filename_actor)
        FS.update_actor(image_actor)
        for i,filename_clbrt in enumerate(list_filenames):
            image_clbrt = cv2.imread(folder_in + filename_clbrt)
            FS.update_clbrt(image_clbrt)

            result = FS.do_faceswap(folder_out=folder_out, do_debug=False)
            cv2.imwrite(folder_out + 'result%02d_%02d.jpg'%(j,i), result)
    return
# ---------------------------------------------------------------------------------------------------------------------
def demo_03(folder_in,folder_out):

    filename_clbrt = folder_in + 'Person5a.jpg'
    image_clbrt = cv2.imread(filename_clbrt)
    FS = tools_faceswap.Face_Swapper(D, image_clbrt, numpy.zeros((10, 10, 3), dtype=numpy.uint8), device='cpu')

    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=cVz12M2awA8','D:/', 'BP')


    #tools_video.extract_frames('D:/BP.mp4', 'D:/BP_in/')
    #FS.process_folder_extract_landmarks('D:/BP_in/', folder_out, write_images=False, write_annotation=True)
    #FS.interpolate(folder_out+'Landmarks.txt',folder_out+'Landmarks_filtered.txt')
    #FS.filter_landmarks(folder_out+'Landmarks_filtered.txt',folder_out+'Landmarks_interpolated.txt')
    #FS.process_folder_faceswap_by_landmarks('D:/BP_in/', folder_out+'Landmarks_BP_interpolated.txt', folder_out)

    #tools_animation.crop_images_in_folder('D:/4_swap/','D:/4_swap_croped/',0, 800, 1080, 1920)
    #tools_animation.folder_to_video(folder_out,'D:/ani_full.mp4',mask='*.jpg',resize_W=1120//2,resize_H=1080//2)

    #tools_animation.merge_images_in_folders('D:/BP_in/','D:/BP_out/','D:/res/',mask='*.jpg')
    #tools_animation.folder_to_video('D:/res/','D:/BP_merged_v02.mp4',mask='*.jpg',resize_W=3840//2,resize_H=1080//2)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = './images/ex_faceswap/01/'
    folder_out = './images/output/'
    list_filenames = tools_IO.get_filenames(folder_in, '*.jpg')
    #list_filenames = ['Person5a.jpg','Person2e.jpg']
    demo_01(folder_out,folder_in + list_filenames[ 0],folder_in + list_filenames[2])
    #demo_02(folder_out,folder_in,list_filenames)
    #demo_03(folder_in,folder_out)


