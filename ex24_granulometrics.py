import argparse
import tools_IO
import processor_Slices
#----------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_bin8_after/'
folder_out = './images/output_ex08/'
# ----------------------------------------------------------------------------------------------------------------------
P = processor_Slices.processor_Slices(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def process_folder_images(folder_in, folder_out):

    #tools_IO.remove_files(folder_out,create=True)
    filenames = tools_IO.get_filenames(folder_in,'*.jpg')

    for filename in filenames[:1]:
        P.process_file_granules(folder_in + filename, do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    process_folder_images(folder_in, folder_out)
    #P.plot_chart_from_folder(folder_out, folder_out)
    #P.plot_separatability('./images/output_ex07/','./images/output_ex08/', folder_out)

    


