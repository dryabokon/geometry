import numpy
import tools_IO
import cv2
# ---------------------------------------------------------------------------------------------------------------------
class Circle:

    def __init__(self, cx, cy, r, label,font_size,icolour=None):
        self.cx, self.cy, self.r,self.label,self.font_size= cx, cy, r,label,font_size
        self.icolour = icolour

    def overlap_with(self, cx, cy, r):
        d = numpy.hypot(cx - self.cx, cy - self.cy)
        return d < r + self.r

    def draw_circle(self, fo):
        print('<circle cx="{}" cy="{}" r="{}" class="c{}"/>'.format(self.cx, self.cy, self.r, self.icolour), file=fo)
        return

    def draw_label(self, fo):
        print('<text x="{}" y="{}" font-size="{}" >{}</text>'.format(self.cx, self.cy,self.font_size,self.label), file=fo)
        return

# ---------------------------------------------------------------------------------------------------------------------
class Circles:

    def __init__(self, labels,weights):
        self.N = len(labels)
        self.width, self.height = 1200, 680
        self.CX, self.CY = self.width // 2, self.height // 2

        self.colours = []
        for i in range(self.N):
            color = cv2.cvtColor(numpy.array([int(255 * i // self.N), 255, 225], dtype=numpy.uint8).reshape(1, 1, 3),cv2.COLOR_HSV2BGR)[0][0]
            self.colours.append('#%02x%02x%02x' % (color[0], color[1], color[2]))

        self.circles = []

        r = (weights - weights.min()) / weights.max()
        r = 10 + r* 60

        font_size = (weights - weights.min()) / weights.max()
        font_size = 8 +font_size*12


        idx = numpy.argsort(-r)
        self.R = 4*r[idx[0]]

        self.colours = numpy.array(self.colours)
        self.colours = self.colours[idx]

        for i in range(self.N):
            self._place_circle(r[idx[i]],labels[idx[i]],font_size[idx[i]],numpy.random.randint(len(self.colours)))
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def preamble(self):
        print('<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n' + ' '*5 +'xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" >'.format(self.width, self.height), file=self.fo)

    # ---------------------------------------------------------------------------------------------------------------------
    def defs_decorator(func):
        def wrapper(self):
            print("""
            <defs>
            <style type="text/css"><![CDATA[""", file=self.fo)

            func(self)

            print("""]]></style>
            </defs>""", file=self.fo)
        return wrapper
    # ---------------------------------------------------------------------------------------------------------------------
    @defs_decorator
    def svg_styles(self):
        """Set the SVG styles: circles are coloured with no border."""

        print('circle {stroke: none;}', file=self.fo)
        for i, c in enumerate(self.colours):
            print('.c{} {{fill: {};}}'.format(i, c), file=self.fo)

    # ---------------------------------------------------------------------------------------------------------------------
    def make_svg(self, filename):
        with open(filename, 'w') as self.fo:
            self.preamble()
            self.svg_styles()
            for circle in self.circles:
                circle.draw_circle(self.fo)

            for circle in self.circles:
                circle.draw_label(self.fo)

            print('</svg>', file=self.fo)
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def _place_circle(self, radius, label,font_size,icolor):

        guard = 500
        while guard:

            cr, cphi = (self.R * numpy.sqrt(numpy.random.random()),2 * numpy.pi * numpy.random.random())
            cx, cy = cr * numpy.cos(cphi), cr * numpy.sin(cphi)
            if cr+radius < self.R:

                if not any(circle.overlap_with(self.CX+cx, self.CY+cy, radius) for circle in self.circles):

                    #circle = Circle(cx + self.CX, cy + self.CY, radius, label, font_size,icolour=numpy.random.randint(len(self.colours)))
                    #xxx = numpy.random.randint(len(self.colours))
                    circle = Circle(cx + self.CX, cy + self.CY, radius, label, font_size,icolour=icolor)
                    self.circles.append(circle)
                    return
            guard -= 1

        print('guard reached.')
        return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    filename_in  = './images/ex_circles/data.txt'
    filename_out = './images/output/circles.svg'
    A = tools_IO.load_mat(filename_in,delim=',')
    weights = numpy.array(A[:, 0], dtype=numpy.int)
    labels = numpy.array(A[:, 1], dtype=numpy.str)



    circles = Circles(labels,weights)
    circles.make_svg(filename_out)