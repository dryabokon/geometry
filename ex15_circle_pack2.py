import numpy
import matplotlib.pyplot as plt
import cv2
# ---------------------------------------------------------------------------------------------------------------------
# create 10 circles with different radii
r = numpy.random.randint(5, 15, size=10)
# ---------------------------------------------------------------------------------------------------------------------
class C():
    def __init__(self,r):
        self.N = len(r)
        self.x = numpy.ones((self.N, 3))
        self.x[:,2] = r
        maxstep = 2*self.x[:,2].max()
        length = numpy.ceil(numpy.sqrt(self.N))
        grid = numpy.arange(0, length * maxstep, maxstep)
        gx,gy = numpy.meshgrid(grid, grid)
        self.x[:,0] = gx.flatten()[:self.N]
        self.x[:,1] = gy.flatten()[:self.N]
        self.x[:,:2] = self.x[:,:2] - numpy.mean(self.x[:, :2], axis=0)
        self.label = []
        self.color = []
        for i in range(self.N):
            self.label.append(' ')
            color = cv2.cvtColor(numpy.array([int(255 * numpy.random.rand()), 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0][0]
            self.color.append([int(color[0]), int(color[1]), int(color[2])])

        self.step = self.x[:,2].min()
        self.p = lambda x,y: numpy.sum((x ** 2 + y ** 2) ** 2)
        self.E = self.energy()
        self.iter = 1.

    def minimize(self):
        while self.iter < 1000*self.N:
            for i in range(self.N):
                rand = numpy.random.randn(2) * self.step / self.iter
                self.x[i,:2] += rand
                e = self.energy()
                if (e < self.E and self.isvalid(i)):
                    self.E = e
                    self.iter = 1.
                else:
                    self.x[i,:2] -= rand
                    self.iter += 1.

    def energy(self):
        return self.p(self.x[:,0], self.x[:,1])

    def distance(self,x1,x2):
        return numpy.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) - x1[2] - x2[2]

    def isvalid(self, i):
        for j in range(self.N):
            if i!=j:
                if self.distance(self.x[i,:], self.x[j,:]) < 0:
                    return False
        return True

    def plot(self, ax):
        for i in range(self.N):
            cc = self.color[i]
            circ = plt.Circle(self.x[i,:2],self.x[i,2],color=[cc[0]/255,cc[1]/255,cc[2]/255],edgecolor=[0.5,0.5,0.5])
            plt.text(self.x[i,0],self.x[i,1], self.label[i],horizontalalignment="center", verticalalignment="center",fontsize=9)
            ax.add_patch(circ)


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    c = C(r)
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    ax.axis("off")
    c.minimize()
    c.plot(ax)
    ax.relim()
    ax.autoscale_view()
    plt.show()