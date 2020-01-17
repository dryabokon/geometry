import squarify
import numpy
import tools_IO
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
<<<<<<< HEAD
def exmaple01():
    filename_in  = './images/ex_circles/data.txt'
=======
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
if __name__ == '__main__':
    filename_in = './images/ex_circles/data.txt'
>>>>>>> 2bc923d03802b228638db0b65c032af8c4a9bd63
    filename_out = './images/output/pack.png'
    A = tools_IO.load_mat(filename_in, delim=',')
    weights = numpy.array(A[:, 0], dtype=numpy.int)
    labels = numpy.array(A[:, 1], dtype=numpy.str)
    labels = [label.replace(' ', '\n') for label in labels]

<<<<<<< HEAD
    N = len(labels)

    cmap = plt.cm.Set3
    colors = numpy.array([cmap(i / float(len(labels))) for i in range(N)])
    colors = colors[numpy.random.choice(N, N,replace=False)]

    fig = plt.figure(figsize=(12, 6))
    squarify.plot(sizes=weights, label=labels,color=colors)
    #plt.tick_params(axis='off', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',labelright='off', labelbottom='off')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename_out)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    exmaple01()


=======
    colors = generate_colors2(len(labels))

    fig = plt.figure(figsize=(8, 6))
    squarify.plot(sizes=weights, label=labels, color=colors)
    plt.tick_params(axis='off', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename_out)
    return
>>>>>>> 2bc923d03802b228638db0b65c032af8c4a9bd63
