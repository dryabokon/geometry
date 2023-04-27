import os
import cv2
import numpy
import math
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import tools_video
import tools_animation
import tools_image
import tools_draw_numpy
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def promo_script0():

    folder_in  = 'D:/Projects/GL/MLOps/media3/'
    folder_out = 'D:/Projects/GL/MLOps/media4/'
    # tools_IO.remove_folders(folder_out)
    # tools_IO.remove_files(folder_out)

    n_frames = 18

    # tools_IO.copy_folder(folder_in + '00_header/', folder_out + '00_header/')
    # tools_IO.copy_folder(folder_in + '01_concept/', folder_out + '01a_concept/')
    tools_animation.fly_effetct(folder_in + '01_concept/', folder_out + '01b_concept/',left=0,top=150,right=1920,bottom=990,n_frames=n_frames,effect='out')
    tools_animation.fly_effetct(folder_in + '02_UI/', folder_out + '02a_UI/', left=0, top=150, right=1920,bottom=990, n_frames=n_frames, effect='in')
    # tools_IO.copy_folder(folder_in + '02_UI/', folder_out + '02b_UI/')
    tools_animation.fly_effetct(folder_in + '02_UI/', folder_out + '02c_UI/', left=0, top=150, right=1920, bottom=990,n_frames=n_frames, effect='out')
    #
    # tools_animation.fly_effetct(folder_in + '03_comp_diagram/', folder_out + '03a_comp_diagram/', left=0, top=150, right=1920, bottom=250,n_frames=n_frames, effect='in')
    # tools_IO.copy_folder(folder_in + '03_comp_diagram/', folder_out + '03b_comp_diagram/')
    # tools_animation.fly_effetct(folder_in + '03_comp_diagram/', folder_out + '03c_comp_diagram/', left=0, top=250,right=1920, bottom=990, n_frames=n_frames, effect='out')
    # tools_IO.copy_folder(folder_in + '04_gcp2/', folder_out + '04a_gcp2/')
    # tools_animation.fly_effetct(folder_out+ '04a_gcp2/',folder_out + '04b_gcp2/', left=0, top=250, right=1920,bottom=990, n_frames=n_frames, effect='out')
    #
    # tools_IO.copy_folder(folder_in + '05_pairplot/', folder_out + '05a_pairplot/')
    # tools_IO.copy_folder(folder_in + '06_auc/', folder_out + '06a_auc/')
    # tools_IO.copy_folder(folder_in + '07_ci/', folder_out + '07a_ci/')
    #
    # tools_animation.fly_effetct(folder_in  + '07_ci/', folder_out + '07b_ci/', left=0, top=250, right=1920,bottom=990, n_frames=n_frames, effect='out')
    # tools_animation.fly_effetct(folder_out + '07b_ci/', folder_out + '07c_ci/', left=0, top=150, right=1920, bottom=250,n_frames=n_frames, effect='out')

    # tools_animation.fly_effetct(folder_in + '08_arch_black/', folder_out + '08a_arch_black/', left=0, top=150, right=1920, bottom=250,n_frames=n_frames, effect='in')
    # tools_IO.copy_folder(folder_in + '08_arch_black/', folder_out + '08b_arch_black/')
    # tools_animation.fly_effetct(folder_in + '08_arch_black/', folder_out + '08c_arch_black/', left=0, top=250, right=1920, bottom=990,n_frames=n_frames, effect='out')
    #
    # tools_animation.fly_effetct(folder_in + '09_maturity_black/', folder_out + '09a_maturity_black/', left=0, top=250, right=1920,bottom=990, n_frames=n_frames, effect='in')
    # tools_IO.copy_folder(folder_in + '09_maturity_black/', folder_out + '09b_maturity_black/')
    # tools_IO.copy_folder(folder_in + '99_footer/', folder_out + '99_footer/')

    #tools_animation.folders_to_video(folder_out,folder_out + 'video.mp4')

    return
# ----------------------------------------------------------------------------------------------------------------------
def promo_script():
    folder_in  = 'D:/Projects/GL/Sugar/media/01/'
    folder_out = 'D:/Projects/GL/Sugar/media/02/'
    # tools_IO.remove_folders(folder_out)
    # tools_IO.remove_files(folder_out)

    n_frames = 18

    tools_animation.folder_to_animated_gif_imageio(folder_in+'scene_00z/', folder_out+'scene_01a_intro/', mask='*.png', framerate=24,stop_ms=0,duration_ms=4000)
    # tools_animation.folder_to_animated_gif_imageio(folder_in+'scene_01/', folder_out+'scene_01b_main/', mask='*.png', framerate=24,stop_ms=500,duration_ms=8000)
    # tools_animation.fly_effetct(folder_in+'scene_01/', folder_out+'scene_01c_out/', left=0, top=150, right=1920, bottom=1080,n_frames=n_frames, effect='out')
    #
    # tools_animation.folder_to_animated_gif_imageio(folder_in + 'scene_02/', folder_out + 'scene_02b_main/', mask='*.png',framerate=24, stop_ms=1000, duration_ms=13000)
    # tools_animation.fly_effetct(folder_in + 'scene_02/', folder_out + 'scene_02b_out/', left=0, top=150, right=1920,bottom=1080, n_frames=n_frames, effect='out')
    #
    # tools_animation.folder_to_animated_gif_imageio(folder_in + 'scene_03/', folder_out + 'scene_03b_main/',mask='*.png', framerate=24, duration_ms=10000)
    #
    # tools_animation.folder_to_animated_gif_imageio(folder_in + 'scene_04/', folder_out + 'scene_04b_main/', mask='*.png',framerate=24, stop_ms=1000, duration_ms=10000)
    # tools_animation.fly_effetct(folder_in + 'scene_04/', folder_out + 'scene_04b_out/', left=0, top=150, right=1920,bottom=1080, n_frames=n_frames, effect='out')
    #
    # tools_animation.folder_to_animated_gif_imageio(folder_in + 'scene_05/', folder_out + 'scene_05b_main/', mask='*.png',framerate=24, stop_ms=6000, duration_ms=9000)
    # tools_animation.fly_effetct(folder_in + 'scene_05/', folder_out + 'scene_05b_out/', left=0, top=150, right=1920,bottom=1080, n_frames=n_frames, effect='out')

    tools_animation.folder_to_animated_gif_imageio(folder_in + 'scene_06/', folder_out + 'scene_06b_main/', mask='*.png',framerate=24, stop_ms=12000, duration_ms=15000)
    tools_animation.fly_effetct(folder_in + 'scene_06/', folder_out + 'scene_06b_out/', left=0, top=150, right=1920,bottom=1080, n_frames=n_frames, effect='out')

    tools_IO.copy_folder(folder_in + 'scene_00_header/', folder_out + 'scene_00_header/')
    tools_IO.copy_folder(folder_in + 'scene_99_footer/', folder_out + 'scene_99_footer/')

    tools_animation.folders_to_video(folder_out,folder_out + 'video.mp4')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #filename_in = 'D:/Projects/VFS/Cashiers 1_IP169_port/02.07.2019 14_59_59 (UTC+03_00).mkv'
    #folder_out = 'D:/Projects/VFS/output/'

    #tools_video.extract_frames_v2('D:/Projects/GL/Hire/ffmpeg/pre_prod/01-Dima.mp4','D:/ccc/')
    #tools_animation.crop_images_in_folder('D:/CCC/','D:/CCC2/',279,58, 918,1480, mask='*.png')
    #tools_animation.folder_to_animated_gif_imageio('D:/ccc2/', 'D:/resources.gif', mask='*.jpg,*.png',stop_ms=3000, framerate=4,do_reverce=True)
    #tools_animation.folder_to_video('D:/ccc/', 'D:/lps.mp4', mask='*.jpg,*.png', framerate=24)

    # folder_in = 'D:/Projects/GL/Hire/01/'
    # folder_out = 'D:/Projects/GL/Hire/output/'
    # tools_animation.fly_effetct(folder_in, folder_out, left=116, top=240, right=1920, bottom=430, n_frames=36, effect='in')
    # tools_animation.folder_to_video(folder_out, folder_out+'Skill-matrix.mp4', mask='*.png', resize_W=1280, resize_H=720,framerate=30,stride=2,stop_ms=1000)

    #tools_animation.re_encode_folder('D:/Projects/GL/Hire/ffmpeg/pre_prod1/', 'D:/Projects/GL/Hire/ffmpeg/pre_prod2/')
    #tools_animation.merge_videos_ffmpeg('D:/Projects/GL/Hire/ffmpeg/pre_prod2/', '*.mp4')
    #tools_video.extract_frames('D:/Projects/GL/Sugar/images/0563.gif','D:/Projects/GL/Sugar/media/01/scene_03b/',prefix='',start_time_sec=0,end_time_sec=None,stride=1,scale=1)

    tools_animation.merge_images_in_folders('./images/out1/', './images/out2/', './images/output/', mask='*.png,*.jpg')
    tools_animation.folder_to_animated_gif_imageio('./images/output/', './images/output/render.gif', mask='*.jpg,*.png',stop_ms=3000, resize_H=2160//4, resize_W=1290//4,framerate=1,do_reverce=False)

