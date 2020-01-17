import squarify
import numpy
import tools_IO
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
def exmaple01():
    filename_in  = './images/ex_circles/data.txt'
    filename_out = './images/output/pack.png'
    A = tools_IO.load_mat(filename_in,delim=',')
    weights = numpy.array(A[:, 0], dtype=numpy.int)
    labels = numpy.array(A[:, 1], dtype=numpy.str)
    #labels = A[:, 1]
    labels = [label.replace(' ', '\n') for label in labels]

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


