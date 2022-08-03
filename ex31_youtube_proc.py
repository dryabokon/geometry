# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=zpYfsmK7aEw','D:/','YS.mp4')
    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=-XPuWK-TBMo','./data/output/','res.mp4')


    #tools_video.extract_frames('D://obhama_fake.mp4','D://1/')
    #tools_animation.folder_to_animated_gif_imageio('D://2/', 'D://fpam.gif', mask='*.jpg', framerate=10,resize_H=720//4, resize_W=1280//4,do_reverce=False)

    #tools_video.extract_frames('D://soccer_dataset.mp4','D://soccer/')
    #tools_animation.crop_images_in_folder('D://1/','D://2/',top=114, left=7, bottom=565, right=1270,mask='*.jpg')
    tools_animation.folder_to_animated_gif_imageio('./images/output/', './images/output/interpolation.gif', mask='*.png', framerate=10,do_reverce=True)


