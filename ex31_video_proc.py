import cv2
import numpy
import math
# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
def cap_06():
    URL = 'https://www.youtube.com/watch?v=ulpIL6KhD50'
    out_path = 'D:/'
    out_filename = 'res'
    tools_video.grab_youtube_video(URL, out_path, out_filename)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #filename_in = 'D:/Projects/VFS/Cashiers 1_IP169_port/02.07.2019 14_59_59 (UTC+03_00).mkv'
    #folder_out = 'D:/Projects/VFS/output/'

    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=g-skPkW75mQ','D:/','dashcam.avi')
    #tools_video.extract_frames_v2('D:/Projects/GL/Melco_CityHealth/Front-V2022-05-11-07-37-50.mp4','D:/ccc/')

    #tools_animation.merge_images_in_folders('D:/ccc3/','D:/ccc2/','D:/ccc4/')
    #tools_animation.folder_to_animated_gif_imageio('D:/ccc/', 'D:/SLAM.gif', mask='*.jpg,*.png', framerate=24,resize_W=1920//4,resize_H=1080//4,do_reverce=False)


    #tools_animation.merge_images_in_folders_temp('D:/iii/','D:/ooo/','D:/ttt/')


    #tools_animation.crop_images_in_folder('D:/ccc/','D:/ccc1/',187,134, 357,938, mask='*.jpg')
    tools_animation.folder_to_video('C:/Users/Anna/source/digits/geometry/images/output/', 'D:/Front-V2022-05-11-07-37-50-BEV.mp4', mask='*.jpg',framerate=29)
    #tools_animation.folder_to_animated_gif_imageio('D:/ccc2/', 'D:/head_pose.gif', mask='*.png,*.jpg',stop_ms=3000,resize_W=1280//4,resize_H=960//4,stride=3,framerate=18)


