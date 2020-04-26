import tools_soccer_field
import tools_IO
import tools_animation
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_lines3/'
folder_out = './images/output/'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def process_folder(folder_in, folder_out):
    tools_IO.remove_files(folder_out, create=True)
    filenames = tools_IO.get_filenames(folder_in, '*.jpg')

    for filename in filenames:
        SFP.process_view(folder_in+filename,do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #process_folder(folder_in,folder_out)
    #tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'ani.gif', mask='*.png', framerate=15,resize_W=1920//4, resize_H=1080//4)

    SFP.train_on_annotation(folder_in + 'annotation.txt')



