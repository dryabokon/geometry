import cv2
import numpy
import pandas as pd
from collections import Counter
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_DF
import tools_IO
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------
#cmap_names =['viridis', 'plasma', 'inferno', 'magma', 'cividis']
#cmap_names =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#cmap_names =['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper']
#cmap_names =['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
#cmap_names = ['twilight', 'twilight_shifted', 'hsv']
#cmap_names = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
#cmap_names = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
#cmap_names = ['Wistia']*1
#cmap_names = ['Reds']*1
#cmap_names = ['Greens']*5
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ---------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ---------------------------------------------------------------------------------------------------------------------
def word_count(filename_in,delim=','):
    all_words = []

    lines = tools_IO.get_lines(filename_in,delim=delim)
    for line in lines:
        for item in line:
            all_words.append(item)

    wordcount = Counter(all_words)
    C = numpy.array([v for v in wordcount.values()])
    rank = tools_IO.rank(C)
    col_id= 255*rank/max(rank)
    df = pd.DataFrame({'cnt': C,'word': wordcount.keys(),'color_id':col_id})
    df = df.sort_values(by='cnt', ascending=False)[:20]

    return df
# ---------------------------------------------------------------------------------------------------------------------
def ex_01_positions():
    df = pd.read_csv('./images/ex_pack_text/positions.txt', sep='\t')
    P.plot_squarify(df, idx_label=1, idx_size=0, palette='~RdBu', stat='%', alpha=0, filename_out='positions.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_02_words():
    #df = word_count(filename_in = './images/ex_pack_text/skills.txt',delim = ',')
    df = word_count(filename_in = './images/ex_pack_text/cover_letter.txt',delim = ' ')
    #col255 = tools_draw_numpy.get_colors(df.shape[0], colormap='viridis',shuffle=False,interpolate=False)
    P.plot_squarify(df, idx_label=1, idx_size=0, stat='#', filename_out='words.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_03_countries():
    df = pd.read_csv('./images/ex_pack_text/countries.csv', sep=',')
    df = df.sort_values(by='pop',ascending=False)[:13]
    colors =  tools_draw_numpy.values_to_colors(df['gdpPercap'],'warm')
    P.plot_squarify(df, idx_label=0, idx_size=4, colors=colors, stat='', filename_out='countries.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_04_sales():
    df = pd.read_csv('./images/ex_pack_text/sales.csv', sep='\t')
    colors = tools_draw_numpy.values_to_colors(df['sales']/df['calls'], 'warm',)#'~RdBu'
    P.plot_squarify(df, idx_label=2, idx_size=4,colors=colors, stat='', palette='RdBu', filename_out='sales.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out)
    # ex_01_positions()
    #ex_02_words()
    #ex_03_countries()
    ex_04_sales()



