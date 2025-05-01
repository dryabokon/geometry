# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=XRyHWiDkD1c','D:/','XRyHWiDkD1c.mp4')
    #tools_video.grab_youtube_stream('https://www.youtube.com/watch?v=71SNlChW_nA', 'D:/71SNlChW_nA_part11.mp4',total_frames=10000)


    tools_video.extract_frames('D://MOV_0005_stabilized_good.avi','D://ccc/',prefix='frame_')
    #tools_video.extract_frames_ffmpeg('D://voronov.mp4', 'D://ccc/', prefix='frame_')
    #tools_animation.folder_to_animated_gif_imageio('D://ccc/', 'D://telegram.gif', mask='*.jpg',stop_ms=2000,framerate=16,resize_W=810, resize_H=840,stride=2,do_reverce=False)

    #tools_video.extract_frames('D://Naft11.mp4','D://ccc/')
    #tools_animation.crop_images_in_folder('D://ccc/','D://ccc2/',top=0, left=0, bottom=720, right=1280,mask='*.jpg')
    #tools_animation.folder_to_animated_gif_imageio('D://ccc2/', 'D://VW.gif', framerate=16,do_reverce=False,stop_ms=0,resize_H=200,resize_W=375)
    #tools_animation.folder_to_video_simple('D://ccc2//','D://Lane4a.mp4',framerate=30)

    #tools_animation.folder_to_animated_gif_imageio('D://source//digits//Fly//UI_App//output//', 'D://track.gif', framerate=8,do_reverce=False,stop_ms=0,resize_W=1280//4,resize_H=720//4)







