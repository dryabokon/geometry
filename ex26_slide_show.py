import cv2
import numpy
import tools_IO
import tools_render_CV
import tools_image
import tools_animation
import progressbar
# ---------------------------------------------------------------------------------------------------------------------
def create_trajectories(W,H):

    trjs = []

    tgs_LU = numpy.arange(1.25,-1.76,-0.5)
    tgs_RD = numpy.arange(3.5, -3.6, -1)

    for tg_LU,tg_RD in zip(tgs_LU,tgs_RD):
        line_LU = numpy.array((2*W, H/2,-2*W, 4*W*tg_LU+H/2))
        line_RD = numpy.array((2*W, H/2,   0, 2*W*tg_RD+H/2))
        trjs.append((line_LU,line_RD))
    return trjs
# ---------------------------------------------------------------------------------------------------------------------
def draw_objects(image_bg,placeholdes,images):

    for p,im in zip(placeholdes,images):

        target_image_height = int(p[3]-p[1])+1
        target_image_width  = int(p[2]-p[0])+1
        start_row = int(p[1])-1
        start_col = int(p[0])-1

        if target_image_height*target_image_width>0:
            small_image = tools_image.smart_resize(im,target_image_height,target_image_width)
            if small_image.shape[0]>0 and small_image.shape[0]>0:
                image_bg = tools_image.put_image(image_bg,small_image,start_row,start_col)

    return image_bg
# ---------------------------------------------------------------------------------------------------------------------
def create_placeholders(trajectories,t,cycle):

    placeholders = []

    for trajectory in trajectories:

        stop_lu = trajectory[0][2:]
        start_lu = trajectory[0][:2]
        stop_rd = trajectory[1][2:]
        start_rd = trajectory[1][:2]

        start_lu = start_lu/2+stop_lu/2
        start_rd = start_rd/2+stop_rd/2

        lu = start_lu + (stop_lu - start_lu) * (t) / (cycle - 1)
        rd = start_rd + (stop_rd - start_rd) * (t) / (cycle - 1)
        placeholders.append((lu[0],lu[1],rd[0],rd[1]))

    return placeholders
# ---------------------------------------------------------------------------------------------------------------------
def create_placeholders_x2(parents,trajectories):

    placeholders = []

    for parent, trajectory in zip(parents,trajectories):
        width = (parent[2]-parent[0])/2
        line_left  = (parent[2],0,parent[2],100)
        line_right = (parent[2]+width,0,parent[2]+width, 100)

        lu = tools_render_CV.line_intersection(line_left , trajectory[0])
        rd = tools_render_CV.line_intersection(line_right, trajectory[1])

        placeholders.append((lu[0], lu[1], rd[0], rd[1]))

    return placeholders
# ---------------------------------------------------------------------------------------------------------------------
def get_images(folder_in,H,W):
    images=[]
    for filename in tools_IO.get_filenames(folder_in,'*.jpg'):
        image = cv2.imread(folder_in+filename)
        image = tools_image.smart_resize(image,H,W)
        images.append(image)

    return numpy.array(images)
# ---------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_pack/'
folder_out = './images/output/'
W,H = 4*128,4*128
# ---------------------------------------------------------------------------------------------------------------------
def refine_random_values(idx,M):

    the_list = numpy.arange(1,idx.shape[1]-1,1)
    the_list = numpy.delete(the_list,2)


    for i in range(len(idx)):
        for v in the_list:
            is_issue1 = numpy.sum(1 * (idx[i] == idx[i, v])) >= 2
            is_issue2 = (i - 1 >= 0) and (numpy.sum(1*(idx[i-1] == idx[i, v]))>0)
            is_issue3 = (v==3) and (i>0) and numpy.sum(1 * (idx[:,v] == idx[i, v])) >= 2
            while is_issue1 or is_issue2 or is_issue3:
                idx[i][v] = int(M * numpy.random.rand())
                is_issue1 = numpy.sum(1*(idx[i] == idx[i, v]))>=2
                is_issue2 = (i - 1 >= 0) and (numpy.sum(1*(idx[i-1] == idx[i, v]))>0)
                is_issue3 = (v==3) and (i>0) and numpy.sum(1 * (idx[:,v] == idx[i, v])) >= 2


    return idx
# ---------------------------------------------------------------------------------------------------------------------
def generate_idx0(C,N,M):

    idx = (N* numpy.random.rand(C + 5, M)).astype(int)
    idx[:,3] = numpy.arange(0,len(idx))
    idx = refine_random_values(idx,N)
    idx[-5:] = idx[:5]

    return idx
# ---------------------------------------------------------------------------------------------------------------------
def generate_idx(C,N,M):

    idx = numpy.full((C + 5, M),-1).astype(int)
    idx[:C, 3] = numpy.arange(0, C)
    idx[:C, 2] = numpy.arange(C,2*C)
    idx[:C, 4] = numpy.arange(2*C,3*C)

    rnd = 1+idx.max() + numpy.random.randint((N - idx.max()-1), size=(C + 5, M))

    idx[idx<0] = rnd[idx<0]

    idx = idx%N

    #idx = refine_random_values(idx, N)
    idx[-5:] = idx[:5]

    return idx
# ---------------------------------------------------------------------------------------------------------------------
def create_slides():
    tools_IO.remove_files(folder_out)
    trjs = create_trajectories(W, H)
    time_cycle = 32*2
    C = 5

    images = get_images(folder_in, H, W)
    idx = generate_idx(C,len(images),len(trjs))

    bar = progressbar.ProgressBar(max_value=time_cycle * C)
    for time in range(time_cycle * C):
        bar.update(time)
        image = numpy.full((H, 2 * W, 3), 32, dtype=numpy.uint8)
        placeholders = create_placeholders(trjs, time % time_cycle, time_cycle)
        image = draw_objects(image, placeholders, images[idx[time // time_cycle]])


        placeholders_small = placeholders.copy()
        for scale in range(4):
            placeholders_small = create_placeholders_x2(placeholders_small, trjs)
            image = draw_objects(image, placeholders_small, images[idx[scale + 1 + time // time_cycle]])


        cv2.imwrite(folder_out + 'frame_%03d.png' % time, image[:, :-W//4])
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #create_slides()
    tools_animation.folder_to_animated_gif_imageio(folder_out,folder_out+'ani.gif', mask='*.png', framerate=12,resize_H=512//2, resize_W=896//2,do_reverce=False)
    #tools_animation.folder_to_video(folder_out, folder_out + 'ani.avi', mask='*.png',framerate=18,resize_W=896,resize_H=512)

