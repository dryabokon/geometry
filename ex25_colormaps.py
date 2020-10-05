import cv2
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import matplotlib.pyplot as plt
from matplotlib import cm
# ---------------------------------------------------------------------------------------------------------------------
filename_in = 'D:\\Image109.png'
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
cmap_names =['viridis', 'plasma', 'inferno', 'magma', 'cividis']
#cmap_names =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#cmap_names =['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper']
#cmap_names =['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
#cmap_names = ['twilight', 'twilight_shifted', 'hsv']
#cmap_names = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
#cmap_names = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    image = cv2.imread(filename_in)
    image = tools_image.desaturate_2d(image)
    for cm_name in cmap_names:
        res = tools_image.hitmap2d_to_colormap(image,plt.get_cmap(cm_name))
        cv2.imwrite(folder_out + 'res_%s.png'%cm_name,res)