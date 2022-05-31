import time
import cv2
import pandas as pd
import numpy
import tools_draw_numpy
import packcircles
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
from CV import tools_Ellipse
# ---------------------------------------------------------------------------------------------------------------------
class Packer:
    tol_away = 18
    min_size = 10
    max_objects = 12
    draw_mode = 'circles'
    cmap = 'tab20'
    W = 720
    H = 720

    def __init__(self,folder_out,dark_mode=False):
        numpy.random.seed(11)
        self.dark_mode = dark_mode
        self.col_bg = (43, 43, 43) if self.dark_mode else (232, 240, 244)
        self.folder_out = folder_out
        self.E = tools_Ellipse.Ellipse_Processor(folder_out)
        self.dct_color={}
        self.scale_factor = None

        return
# ---------------------------------------------------------------------------------------------------------------------
    def pretty_print(self,value):

        if   value>1000000:res = '%dM'%int(value/1000000)
        elif value>   1000:res = '%dk'%int(value/1000)
        else:              res = '%d' % value

        return res
# ---------------------------------------------------------------------------------------------------------------------
    def scale_sizes(self,sizes):
        if self.scale_factor is None:
            self.scale_factor = sizes.max()/100

        sizes_scaled = sizes/self.scale_factor
        return sizes_scaled
# ---------------------------------------------------------------------------------------------------------------------
    def filter_visible(self):

        sizes_scaled = self.scale_sizes(self.sizes)
        idx_sorted = numpy.argsort(-self.sizes)[:self.max_objects]

        sizes_scaled = sizes_scaled[idx_sorted].astype(numpy.int)
        labels = self.labels[idx_sorted]
        descriptions = self.descriptions[idx_sorted]

        idx = sizes_scaled >= self.min_size
        return sizes_scaled[idx], labels[idx], descriptions[idx]
# ---------------------------------------------------------------------------------------------------------------------
    def do_pack(self,sizes):
        centers = numpy.array([c for c in packcircles.pack((sizes))])[:, :2]
        xx = numpy.dot(centers[:, 0], sizes) / sizes.sum()
        yy = numpy.dot(centers[:, 1], sizes) / sizes.sum()
        centers += numpy.array((-xx + self.W / 2, -yy + self.H / 2))
        return centers
# ---------------------------------------------------------------------------------------------------------------------
    def init(self,sizes,labels,descriptions):
        self.sizes = sizes
        self.labels = labels
        self.descriptions = descriptions
        self.idx_sizes = numpy.argsort(self.sizes)

        N = self.sizes.shape[0]
        colors = 255 * (tools_draw_numpy.get_colors(N, colormap=self.cmap, alpha_blend=0.0, clr_blend=(0, 0, 0),shuffle=True) / 255)
        colors = colors.astype(int)

        for color,label in zip(colors,labels):
            if label not in self.dct_color:
                self.dct_color[label] = color

        return
# ---------------------------------------------------------------------------------------------------------------------
    def insert(self,label,size,description):

        colors = 255 * (tools_draw_numpy.get_colors(10, colormap=self.cmap, alpha_blend=0.0, clr_blend=(0, 0, 0),shuffle=True) / 255)
        color = colors[0]

        numpy.insert(self.labels, 0, label)
        numpy.insert(self.sizes, 0, size)
        numpy.insert(self.descriptions, 0, description)
        self.dct_color[label] = color

        return
# ---------------------------------------------------------------------------------------------------------------------
    def update(self, sizes, labels, descriptions):

        for label,size,description in zip(labels,sizes,descriptions):
            i = numpy.where(self.labels==label)[0]
            if len(i)>0:
                self.sizes[i] = size
                self.descriptions[i] = description
            else:
                self.insert(label,size,description)
                xx=0

        del_lst = []
        for i,label in enumerate(self.labels):
            if len(numpy.where(labels == label)[0])==0:
                del_lst.append(i)

        if len(del_lst)>0:
            numpy.delete(self.labels,del_lst)
            numpy.delete(self.sizes, del_lst)
            numpy.delete(self.descriptions, del_lst)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def save(self,filename_out):

        df = pd.DataFrame(
            {
                'c_X': self.centers[:,0],
                'c_Y': self.centers[:,1],
                'sizes': self.sizes,
                'labels': self.labels,
                'color_R': self.colors[:,0],
                'color_G': self.colors[:,1],
                'color_B': self.colors[:,2],
            },
            index=None)

        df = df.astype({'labels': 'string'})
        df.to_csv(self.folder_out+filename_out,sep='\t')

        return
# ---------------------------------------------------------------------------------------------------------------------
    def load(self,filename_in):

        df = pd.read_csv(self.folder_out+filename_in,sep='\t')
        df = df.astype({'labels': 'string'})

        self.centers=numpy.hstack((df['c_X'].to_numpy(),df['c_Y'].to_numpy()))
        self.sizes = df['sizes'].to_numpy()
        self.labels = df['labels'].to_numpy()
        self.colors=numpy.concatenate((df['color_R'].to_numpy(),df['color_G'].to_numpy(),df['color_B'].to_numpy()))

        return
# ---------------------------------------------------------------------------------------------------------------------
    def is_pair_separated(self, c1, c2, r1, r2):

        delta = numpy.array(c1-c2)
        if numpy.linalg.norm(delta)< self.tol_away:
            return False

        overlap = r1+r2+self.tol_away-numpy.linalg.norm(delta)
        is_away = overlap<0
        return is_away
# ---------------------------------------------------------------------------------------------------------------------
    def move_away_pair(self, c1,c2,r1,r2,froze_first=False):

        delta = numpy.array(c1 - c2)

        if numpy.linalg.norm(delta)>1:

            overlap = r1 + r2 + self.tol_away - numpy.linalg.norm(delta)
            direction = overlap * delta / numpy.linalg.norm(delta)
            if froze_first:
                d1 = numpy.array((0,0))
                d2 = direction
            else:
                d1 = numpy.array(direction * r2 / (r1 + r2))
                d2 = numpy.array(direction * r1 / (r1 + r2))

            c1+=d1
            c2-=d2

        else:
            if r1>r2:
                c2 += numpy.array((self.tol_away + r1+r2 - numpy.linalg.norm(delta), 0))
            else:
                c2 += numpy.array((self.tol_away + r1+r2 - numpy.linalg.norm(delta), 0))

        return c1,c2
# ---------------------------------------------------------------------------------------------------------------------
    def get_gravity_force_and_direction(self, i, j):

        c1 = numpy.array(self.centers[i])
        c2 = numpy.array(self.centers[j])
        r1 = self.sizes[i]
        r2 = self.sizes[j]

        delta = numpy.array(c1 - c2)
        overlap = r1 + r2 + self.tol_away - numpy.linalg.norm(delta)
        if overlap > 0: return 0,numpy.array((0,0))

        direction_one = delta/numpy.linalg.norm(delta)
        F = (r1**2)*(r2**2)/(numpy.linalg.norm(delta)**2)
        return F,direction_one
# ---------------------------------------------------------------------------------------------------------------------
    def get_distances(self,centers,sizes):

        N = centers.shape[0]
        D = []
        for i in range(N - 1):
            c1 = centers[i]
            for j in range(i+1, N):
                c2 = centers[j]
                delta = numpy.array(c1 - c2)
                distance = numpy.linalg.norm(delta)- sizes[i] - sizes[j]
                D.append(distance)

        D = numpy.array(D)

        return D
# ---------------------------------------------------------------------------------------------------------------------
    def move_away_all(self,centers,sizes,labels, descriptions):
        froze_first = False
        c =0
        distance_min = self.get_distances(centers,sizes).min()
        idx_sizes = numpy.argsort(sizes)

        while (distance_min<self.tol_away-1):
            c+=1
            for i in idx_sizes:
                for j in idx_sizes:
                    if j==i:continue
                    if self.is_pair_separated(centers[i], centers[j], sizes[i], sizes[j]):continue
                    centers[i], centers[j] = self.move_away_pair(centers[i], centers[j], sizes[i], sizes[j],froze_first)

            distance_min = self.get_distances(centers,sizes).min()

        return centers
# ---------------------------------------------------------------------------------------------------------------------
    def get_position_sign(self,sign,font_size):
        W,H = 640,480
        image = numpy.zeros((H, W, 3), dtype=numpy.uint8)
        image = tools_draw_numpy.draw_text(image, sign, (int(10), int(H//2)),font_size=int(font_size),color = (255,255,255))

        #image = cv2.putText(image, '{0:s}'.format(sign), (int(10), int(H//2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),1, cv2.LINE_AA)
        #image = cv2.rectangle(image, (min_x,min_y), (max_x,max_y), (255,0,0))
        #cv2.imwrite(folder_out+'xxx.png',image)

        A = numpy.max(image[:, :, 0], axis=0)
        NZ = numpy.nonzero(A)
        min_x = NZ[0].min()
        max_x = NZ[0].max()

        A = numpy.max(image[:, :, 0], axis=1)
        NZ = numpy.nonzero(A)
        min_y = NZ[0].min()
        max_y = NZ[0].max()

        sx = max_x - min_x
        sy = max_y - min_y
        shift_x = -sx / 2 - min_x + 10
        shift_y = -sy / 2 - min_y + int(H//2)
        return shift_x,shift_y
# ---------------------------------------------------------------------------------------------------------------------
    def plot(self,centers, sizes, labels, descriptions,filename_out=None):

        image = numpy.full((self.H,self.W,3),self.col_bg,dtype=numpy.uint8)

        for i in numpy.argsort(sizes):
            center, radius, label, description = centers[i], sizes[i], labels[i], descriptions[i]
            color = self.dct_color[label]
            p = (int(center[0] - radius), int(center[1] - radius), int(center[0] + radius), int(center[1] + radius))
            if self.draw_mode == 'circles':
                image = tools_draw_numpy.draw_circle_aa(image, int(center[1]), int(center[0]), int(radius), color,clr_bg=self.col_bg)
                image = tools_draw_numpy.draw_circle(image, center[1],center[0], radius, color, alpha_transp=0.6)
            elif self.draw_mode == 'rects':
                image = tools_draw_numpy.draw_rect(image,p[0],p[1], p[2], p[3], color.tolist(), w=1, alpha_transp=0.5)
            else:
                p = (int(center[0] - radius), int(center[1] - radius), int(center[0] + radius), int(center[1] + radius))
                image = tools_draw_numpy.draw_ellipse0(image, p, color, transperency=0.5)

            if description is not None:
                font_size = int(radius)
                shift_x,shift_y = self.get_position_sign(description,font_size)
                image = tools_draw_numpy.draw_text(image,description,(int(center[0]+shift_x), int(center[1]+shift_y)),color,font_size)

        if filename_out is not None:
            cv2.imwrite(self.folder_out + filename_out,image)

        return
# ----------------------------------------------------------------------------------------------------------------------
filename_in  = './images/ex_pack_text/positions.txt'
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
Packer = Packer(folder_out=folder_out)
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out, '*.png')
    df = pd.read_csv('./images/ex_pack_text/population-density.csv',sep=',')

    years = numpy.sort(numpy.unique(df['Year'].to_numpy()))
    for year in years:
        df_temp = df[df.Year==year]
        labels = df_temp.Entity.to_numpy()
        sizes = df_temp.iloc[:,-1].to_numpy()
        descriptions = numpy.array(['%s\n%s' % (label, Packer.pretty_print(size)) for label, size in zip(labels, sizes)])
        sizes = numpy.sqrt(sizes)

        if year == years[0]:
            Packer.init(sizes, labels, descriptions)
            sizes_vis, labels_vis, descriptions_vis = Packer.filter_visible()
            centers_vis = Packer.do_pack(sizes_vis+Packer.tol_away)
            Packer.plot(centers_vis, sizes_vis, labels_vis, descriptions_vis, 'Y_%04d.png' % year)
        else:
            Packer.update(sizes, labels, descriptions)
            sizes_vis_new, labels_vis_new, descriptions_vis_new = Packer.filter_visible()
            centers_vis_prev = []
            for label in labels_vis_new:
                if numpy.array((labels_vis==label)).sum()>0:
                    i = numpy.where(labels_vis==label)
                    centers_vis_prev.append(centers_vis[i[0][0]])
                else:
                    centers_vis_prev.append(numpy.array((Packer.W/2,Packer.H/2)))

            centers_vis_new = Packer.move_away_all(numpy.array(centers_vis_prev),sizes_vis_new,labels_vis_new, descriptions_vis_new)
            Packer.plot(centers_vis_new, sizes_vis_new, labels_vis_new, descriptions_vis_new, 'Y_%04d.png' % year)
            centers_vis, sizes_vis, labels_vis, descriptions_vis = centers_vis_new.copy(), sizes_vis_new.copy(),labels_vis_new.copy(),descriptions_vis_new.copy()

        print(year)


