import squarify
import numpy
import tools_IO
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
def generate_colors(N):
    cmap = plt.cm.Set2
    colors = numpy.array([cmap(i/len(labels)) for i in range(N)])
    #colors = colors[numpy.random.choice(N, N, replace=False)]
    return colors
# ---------------------------------------------------------------------------------------------------------------------
def generate_colors2(N):
    cmap = plt.cm.Paired
    colors = []
    M = len(cmap.colors)

    for i in range(N):
        i1 = (M-1)*i//(N-1)
        alpha = float((M-1)*i/(N-1)) - i1
        colors.append(numpy.array(cmap(i1)) * (1-alpha) + numpy.array(cmap(i1 + 1)) * (alpha))

    #colors = numpy.array(colors)[numpy.random.choice(N, N, replace=False)]

    return colors

# ---------------------------------------------------------------------------------------------------------------------
#cmap_names =['viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmap_names =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#cmap_names =['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper']
#cmap_names =['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
#cmap_names = ['twilight', 'twilight_shifted', 'hsv']
#cmap_names = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
#cmap_names = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    filename_in = './images/ex_circles/data.txt'
    filename_out = './images/output/pack.png'
    folder_out = './images/output/'
    A = tools_IO.load_mat(filename_in, delim=',')
    weights = numpy.array(A[:, 0], dtype=numpy.int)
    labels = numpy.array(A[:, 1], dtype=numpy.str)
    labels = [label.replace(' ', '\n') for label in labels]

    N = len(labels)

    for cmap_name in cmap_names:

        cmap = plt.get_cmap(cmap_name)
        colors = numpy.array([cmap(i / float(len(labels))) for i in range(N)])
        colors = colors[numpy.random.choice(N, N,replace=False)]

        fig = plt.figure(figsize=(12, 6))
        squarify.plot(sizes=weights, label=labels,color=colors)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(folder_out + 'res_%s.png'%cmap_name)
        plt.clf()


