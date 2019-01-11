import numpy
import os
from os import listdir
import fnmatch
from shutil import copyfile
import shutil
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
#from skimage.filters import threshold_mean,threshold_otsu, threshold_adaptive,threshold_local
#from sklearn.externals import joblib
from scipy.misc import toimage
from PIL import Image
import re
#from skimage.morphology import skeletonize
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def smart_index(array, value):
    return numpy.array([i for i, v in enumerate(array) if (v == value)])

# ----------------------------------------------------------------------------------------------------------------------
def remove_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
# ----------------------------------------------------------------------------------------------------------------------
def remove_files(path):

    if not os.path.exists(path):
        return

    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        if os.path.isdir(path + f):
            # shutil.rmtree(path + f)
            continue
        else:
            os.remove(path + f)
    return
# ----------------------------------------------------------------------------------------------------------------------
def remove_folders(path):

    if (path==None):
        return

    if not os.path.exists(path):
        return

    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        if os.path.isdir(path + f):
            shutil.rmtree(path + f)
    return
# ----------------------------------------------------------------------------------------------------------------------
def save_arrays_as_images(path, cols, rows, array, labels=None, filenames=None,descriptions=None):

    if not os.path.exists(path):
        os.makedirs(path)

    if descriptions is not None:
        f_handle = open(path+"descript.ion", "a+")

    if(array.ndim ==2):
        N=array.shape[0]
    else:
        N=1

    for i in range(0,N):

        if(array.ndim == 2):
            arr = array[i]
            if descriptions is not None:
                description = descriptions[i]


            if filenames is not None:
                short_name = filenames[i]
            else:
                short_name = "%s_%05d.bmp" % (labels[i], i)

        else:
            arr = array[i]
            if descriptions is not None:
                description = descriptions[i]

            if filenames is not None:
                short_name = filenames[i]
            else:
                short_name = "%s_%05d.bmp" % (labels, i)


        img= toimage(arr.reshape(rows, cols).astype(int)).convert('RGB')
        img.save(path + short_name)

        if descriptions is not None:
            f_handle.write("%s %s\n" % (short_name, description))

    if descriptions is not None:
        f_handle.close()

    return

# ----------------------------------------------------------------------------------------------------------------------
def save_raw_vec(vec, filename,mode=(os.O_RDWR|os.O_APPEND),fmt='%d',delim=' '):

    if not os.path.isfile(filename):
        mode = os.O_RDWR | os.O_CREAT

    f_handle = os.open(filename,mode)

    s = ""
    for i in range(0, vec.shape[0]-1):
        value = ((fmt+delim) % vec[i]).encode()
        os.write(f_handle,value)

    value = ((fmt+'\n') % vec[vec.shape[0]-1]).encode()
    os.write(f_handle, value)
    os.close(f_handle)

    return
# ----------------------------------------------------------------------------------------------------------------------
def save_mat(mat, filename,fmt='%d',delim=' '):
    numpy.savetxt(filename, mat,fmt=fmt,delimiter=delim)
    return
# ----------------------------------------------------------------------------------------------------------------------
def save_data_to_feature_file_float(filename,array,target):

    m = numpy.array(array).astype('float32')
    v = numpy.matrix(target).astype('float32')
    mat = numpy.concatenate((v.T,m),axis=1)
    #print(mat)
    numpy.savetxt(filename, mat, fmt='%+2.2f',delimiter='\t')
    return

# ----------------------------------------------------------------------------------------------------------------------
def count_lines(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    f.close()

    return lines
# ----------------------------------------------------------------------------------------------------------------------
def load_mat(filename, dtype=numpy.int, delim=' ', lines=None):
    mat  = numpy.genfromtxt(filename, dtype=dtype, delimiter=delim)
    return mat
# ----------------------------------------------------------------------------------------------------------------------
def load_mat_var_size(filename,dtype=numpy.int,delim='\t'):
    l=[]
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                l.append(line.split(delim))
    return l
# ----------------------------------------------------------------------------------------------------------------------
def get_column(list_of_list,col):
    res=[]
    for i in range (0,len(list_of_list)):
        if (col<len(list_of_list[i])):
            res.append(list_of_list[i][col])
        else:
            res.append('-')
    return res
# ----------------------------------------------------------------------------------------------------------------------
def remove_column(list_of_list,col):
    res=[]
    for i in range (0,len(list_of_list)):
        lst = list_of_list[i][0:col] + list_of_list[i][col+1::]
        res.append(lst)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def my_print_sting(strng, space=[]):
    if (strng.ndim != 1):
        return

    fm = "%s"

    for j in range(0, strng.shape[0]):
        s = (fm) % strng[j]
        print(space + s, end=' ')
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def my_print_vector(mat):
    if (mat.ndim != 1):
        return

    mx = mat.max()

    fm = "%d"
    if mx >= 10:
        fm = "%2d"
    if mx >= 100:
        fm = "%3d"
    if mx >= 1000:
        fm = "%4d"
    if mx >= 10000:
        fm = "%5d"

    mat.astype(int)

    for j in range(0, mat.shape[0]):
        s = (fm) % mat[j]
        print(s, end=" ")
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def my_print_int(mat, rows=None, cols=None,file = None):

    if (mat.ndim == 1):
        return my_print_vector(mat)

    if (rows is not None):
        l = numpy.array([len(each) for each in rows]).max()
        desc_r = numpy.array([" " * (l - len(each)) + each for each in rows]).astype(numpy.chararray)

    mx = mat.max()

    if (cols is not None):
        mx = max(mx,cols.max())

    fm = "%1d"
    if mx >= 10:
        fm = "%2d"
    if mx >= 100:
        fm = "%3d"
    if mx >= 1000:
        fm = "%4d"
    if mx >= 10000:
        fm = "%5d"


    if (cols is not None):
        if (rows is not None):
            print(" " * len(desc_r[0]) + ' |', end="",file=file)

        for each in cols:
            print((fm) % each, end=" ",file=file)

        print(file=file)
        print("-" * (len(desc_r[0])+2), end="",file=file)
        print("-" * cols.shape[0]*(int(fm[1])+1),file=file)



    mat.astype(int)
    for i in range(0, mat.shape[0]):
        if (rows is not None):
            print(desc_r[i] + ' |', end="",file=file)

        for j in range(0, mat.shape[1]):
            print(((fm) % mat[i, j]), end=" ",file=file)
        print(file=file)

    if (cols is not None):
        print("-" * (len(desc_r[0]) + 2), end="", file=file)
        print("-" * cols.shape[0] * (int(fm[1]) + 1), file=file)

    return

# ----------------------------------------------------------------------------------------------------------------------
def resize_image(image, target_size,mode='L'):

    w,h = image.size
    ratio = w/h


    xxx=image.resize((int(ratio*target_size[1]),target_size[1]))
    w, h = xxx.size

    offset = (int((-w + target_size[0]) / 2), int((-h + target_size[1]) / 2))
    back = Image.new(mode, target_size, "white")
    back.paste(xxx, offset)

    return back
# ----------------------------------------------------------------------------------------------------------------------
def resize_image0(image, max_size):

    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)))
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]))

    offset = (  int((max_size[0] - scaled.size[0]) / 2), int((max_size[1] - scaled.size[1]) / 2))
    back = Image.new("L", max_size, "white")
    back.paste(scaled, offset)
    return back
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def get_sub_folder_from_folder(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir() ]
    return subfolders
# ----------------------------------------------------------------------------------------------------------------------
def count_images_from_folder(path, mask="*.bmp"):

    filenames = []
    i=0

    for image_name in fnmatch.filter(listdir(path), mask) :
        try:
            img = Image.open(path + image_name).convert('L')
            filenames.append(image_name)
        except OSError:
            i=i
            #print("error")

    return len(filenames)
# ----------------------------------------------------------------------------------------------------------------------
def load_aligned_images_from_folder(path, label, mask="*.bmp", exclusion_folder=None, limit=None, resize_W=None,resize_H=None):
    exclusions = []
    if (exclusion_folder != None) and (os.path.exists(exclusion_folder)):

        for each in os.listdir(exclusion_folder):
            exclusions.append(each[:len(each) - 6] + ".bmp")

    images = []
    filenames = []
    i = 0
    img = 0
    for image_name in fnmatch.filter(listdir(path), mask) :
        if ((limit == None) or (i < limit)) and (image_name not in exclusions):
            try:
                img = Image.open(path + image_name).convert('L')
                if ((resize_W is not None) and (resize_H is not None) and((resize_W,resize_H) != img.size)):
                    #img0 = img.resize((resize_W, resize_H))
                    img = resize_image(img,(resize_W,resize_H))
                vec = numpy.array(img).flatten()

                if len(images) == 0:
                    images = vec
                else:
                    images = numpy.vstack((images, vec))
                filenames.append(image_name)
                i = i + 1
            except OSError:
                i=i
                #print("error")


    images = numpy.array(images)
    filenames = numpy.array(filenames)

    labels = numpy.full(images.shape[0], label)
    DX = img.size[0]
    DY = img.size[1]



    return (images, labels, filenames, DX, DY)


# ----------------------------------------------------------------------------------------------------------------------
def print_accuracy(labels_fact, labels_pred,patterns,filename = None):

    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    mat = confusion_matrix(numpy.array(labels_fact).astype(numpy.int), numpy.array(labels_pred).astype(numpy.int))
    accuracy = [100-100*mat[i,i]/numpy.sum(mat[i,:]) for i in range (0,mat.shape[0])]
    TP = numpy.trace(mat)
    idx = numpy.argsort(accuracy).astype(int)

    descriptions = numpy.array([('%s %3d%%' % (patterns[i], 100-accuracy[i])) for i in range (0,mat.shape[0])])

    a_test = numpy.zeros(labels_fact.shape[0])
    a_pred = numpy.zeros(labels_fact.shape[0])

    for i in range(0,a_test.shape[0]):
        a_test[i] = smart_index(idx ,int(labels_fact[i]))[0]
        a_pred[i] = smart_index(idx, int(labels_pred[i]))[0]

    mat2 = confusion_matrix(numpy.array(a_test).astype(numpy.int), numpy.array(a_pred).astype(numpy.int))
    ind = numpy.array([('%3d' % i) for i in range(0, idx.shape[0])])

    l = numpy.array([len(each) for each in descriptions]).max()
    descriptions = numpy.array([" " * (l - len(each)) + each for each in descriptions]).astype(numpy.chararray)
    descriptions = [ind[i] + ' | ' + descriptions[idx[i]] for i in range(0, idx.shape[0])]

    my_print_int(numpy.array(mat2).astype(int),rows=descriptions,cols=ind.astype(int),file = file)

    print("Accuracy = %d/%d = %1.4f" % (TP, numpy.sum(mat), TP / numpy.sum(mat)),file=file)
    print("Fails    = %d" % (numpy.sum(mat) - TP),file=file)
    print(file=file)

    if (filename != None):
        file.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def print_reject_rate(labels_fact, labels_pred, labels_prob,filename=None):

    hit = numpy.array([labels_fact[i] == labels_pred[i] for i in range(0, labels_fact.shape[0])]).astype(int)
    mat = numpy.vstack((labels_prob,hit))
    mat = mat.T
    idx = numpy.argsort(mat[:,0]).astype(int)
    mat2 = numpy.vstack((mat[:,0][idx],mat[:,1][idx]))
    mat2 = mat2.T


    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    decisions,accuracy,th  = [],[],[]

    for i in range(0,mat2.shape[0]):
        dec = mat2.shape[0]-i
        hits = numpy.sum(mat2[i:,1])
        decisions.append(float(dec/mat2.shape[0]))
        #accuracy.append(float(hits/dec))
        accuracy.append(int(100*hits/dec)/100)
        th.append(mat2[i,0])

    decisions2, accuracy2, th2 = [], [], []

    if (filename == None):
        print()
        print()

    print('Dcsns\tAccrcy\tCnfdnc', file=file)
    for each in numpy.unique(accuracy):
        idx = smart_index(accuracy,each)[0]
        print('%1.2f\t%1.2f\t%1.2f' % (decisions[idx],accuracy[idx],th[idx]),file=file)

    if (filename!=None):
        file.close()


    return
# ----------------------------------------------------------------------------------------------------------------------

def print_top_fails(labels_fact, labels_pred, patterns,filename = None):

    if (filename!=None):
        file = open(filename, 'w')
    else:
        file = None

    mat = confusion_matrix(numpy.array(labels_fact).astype(numpy.int), numpy.array(labels_pred).astype(numpy.int))

    error, class1, class2 = [], [], []
    for i in range (0,mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if(i != j):
                error.append(mat[i,j])
                class1.append(i)
                class2.append(j)

    error = numpy.array(error)
    idx = numpy.argsort(-error).astype(int)

    if (filename == None):
        print()
        print('Typical fails:')


    for i in range(0,error.shape[0]):
        if(error[idx[i]]>0):
            print('%3d %s %s' % (error[idx[i]],patterns[class1[idx[i]]],patterns[class2[idx[i]]]),file=file)

    if (filename != None):
        file.close()

    return
# ----------------------------------------------------------------------------------------------------------------------


def split_samples(input_folder, folder_part1, folder_part2, ratio=0.5):
    print("Split samples..")
    if not os.path.exists(folder_part1):
        os.makedirs(folder_part1)
    else:
        remove_files(folder_part1)
        remove_folders(folder_part1)

    if not os.path.exists(folder_part2):
        os.makedirs(folder_part2)
    else:
        remove_files(folder_part2)
        remove_folders(folder_part2)

    folder_list = [f for f in os.listdir(input_folder)]

    for f in folder_list:
        folder_name = input_folder + f
        if os.path.isdir(folder_name):
            os.makedirs(folder_part1 + f)
            os.makedirs(folder_part2 + f)
            print(f)
            file_list = [f for f in os.listdir(folder_name)]
            for file_name in file_list:
                if (random.random() > ratio):
                    copyfile(folder_name + '/' + file_name, folder_part1 + f + '/' + file_name)
                else:
                    copyfile(folder_name + '/' + file_name, folder_part2 + f + '/' + file_name)
    return


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_labels(out_filename, filenames, labels,append=0,delim='\t'):
    if labels.shape[0]!= len(filenames):
        return

    if(append== 0):
        f_handle = open(out_filename, "w")
        f_handle.write("header1%cheader2xx\n" % delim)
    else:
        f_handle = open(out_filename, "a")

    for i in range(0,labels.shape[0]):
        #f_handle.write("%s%c%03d\n" % (filenames[i],delim,labels[i]))
        f_handle.write("%s%c%s\n" % (filenames[i],delim,labels[i]))

    f_handle.close()

    #if append>0:
    #sort_labels(out_filename,delim)

    return


# ----------------------------------------------------------------------------------------------------------------------
def intersection(x11, x12, y11, y12, x21, x22, y21, y22):
    if ((y12 < y21) or (y22 < y11) or (x22 < x11) or (x12 < x21)):
        return 0
    else:
        dx = min(x12, x22) - max(x11, x21)
        dy = min(y12, y22) - max(y11, y21)
        return dx * dy
    return

# ----------------------------------------------------------------------------------------------------------------------
def list_to_chararray(input_list):
    bufer = numpy.array(list(input_list)).astype(numpy.chararray)
    bufer = ''.join(bufer)
    return bufer


# ----------------------------------------------------------------------------------------------------------------------
def plot_tp_fp(plt,fig,tpr,fpr,roc_auc,caption=''):

    #ax = fig.gca()
    #ax.set_xticks(numpy.arange(0, 1.1, 0.1))
    #ax.set_yticks(numpy.arange(0, 1.1, 0.1))

    lw = 2
    plt.plot(fpr, tpr, color='darkgreen', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1.05], [0, 1.05], color='lightgray', lw=lw, linestyle='--')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.set_title(caption + ('AUC = %0.4f' % roc_auc))
    #plt.legend(loc="lower right")
    plt.grid(which='major', color='lightgray', linestyle='--')
    fig.canvas.set_window_title(caption + ('AUC = %0.4f' % roc_auc))
    #plt.show()
# ----------------------------------------------------------------------------------------------------------------------
def plot_multiple_tp_fp(tpr,fpr,roc_auc,desc,caption=''):

    fig = plt.figure()
    fig.canvas.set_window_title(caption)
    ax = fig.gca()
    ax.set_xticks(numpy.arange(0, 1.1, 0.1))
    ax.set_yticks(numpy.arange(0, 1.1, 0.1))

    lw = 2
    plt.plot([0, 1.1], [0, 1.1], color='lightgray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()

    colors_list = list(('lightgray','red','blue','purple','green','orange','cyan','darkblue'))
    lbl = []
    lbl.append('')

    for i in range(0,len(roc_auc)):
        plt.plot(fpr[i], tpr[i],  lw=lw, color=colors_list[i+1],alpha = 0.5)
        lbl.append('%0.2f %s' % (roc_auc[i],desc[i]))

    plt.legend(lbl, loc=4)

    leg = plt.gca().get_legend()
    for i in range(0, len(roc_auc)):
        leg.legendHandles[i].set_color(colors_list[i])

    leg.legendHandles[0].set_visible(False)



# ----------------------------------------------------------------------------------------------------------------------
def get_roc_data_from_scores_file(path_scores):

    data = load_mat(path_scores, numpy.chararray, ' ')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return tpr,fpr,roc_auc
# ----------------------------------------------------------------------------------------------------------------------
def get_roc_data_from_scores_file_v2(path_scores_pos,path_scores_neg):

    data = load_mat(path_scores_pos, numpy.chararray, '\t')
    l1= (data [1:, 0]).astype('float32')
    s1= data[1:, 1:].astype('float32')

    data = load_mat(path_scores_neg, numpy.chararray, '\t')
    l0= (data [1:, 0]).astype('float32')
    s0= data[1:, 1:].astype('float32')

    labels = numpy.hstack((l0, l1))
    scores = numpy.vstack((s0, s1))

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return tpr,fpr,roc_auc
# ----------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_file(plt,fig,path_scores,caption=''):

    data = load_mat(path_scores, dtype = numpy.chararray, delim='\t')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plot_tp_fp(plt,fig,tpr,fpr,roc_auc,caption)

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_scores(plt,fig,filename_data_pos,filename_data_neg,filename_data_grid,filename_scores_grid,th,noice_needed=0,caption=''):

    data = load_mat(filename_scores_grid, dtype=numpy.chararray, delim='\t')[1:,:]
    grid_scores = data[:, 1:].astype('float32')

    data = load_mat(filename_data_grid, dtype=numpy.chararray, delim='\t')
    data_grid = data[:,1:].astype('float32')

    data = load_mat(filename_data_pos, dtype=numpy.chararray, delim='\t')
    l1 = (data[:, 0]).astype('float32')
    x1 = data[:,1:].astype('float32')

    data = load_mat(filename_data_neg, dtype=numpy.chararray, delim='\t')
    l2 = (data[:, 0]).astype('float32')
    x2 = data[:,1:].astype('float32')

    X = numpy.vstack((x1,x2))
    labels = numpy.hstack((l1, l2)).astype(int)

    X1 = X[labels >  0]
    X0 = X[labels <= 0]

    #'''
    max = numpy.max(grid_scores)
    min = numpy.min(grid_scores)
    for i in range(0,grid_scores.shape[0]):
        if(grid_scores[i]>th):
            grid_scores[i]=(grid_scores[i]-th)/(max-th)
        else:
            grid_scores[i] = (grid_scores[i] - th) / (th-min)
    #'''

    S=int(math.sqrt(grid_scores.shape[0]))
    grid_scores=numpy.reshape(grid_scores,(S,S))

    minx=numpy.min(data_grid[:, 0])
    maxx=numpy.max(data_grid[:, 0])
    miny=numpy.min(data_grid[:, 1])
    maxy=numpy.max(data_grid[:, 1])


    if noice_needed>0:
        noice1 = 0.05-0.2*numpy.random.random_sample(X1.shape)
        noice0 = 0.05-0.2*numpy.random.random_sample(X0.shape)
        X1+=noice1
        X0+=noice0

    plt.set_title(caption)

    xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=S), numpy.linspace(miny, maxy,num=S))

    plt.contourf(xx, yy, numpy.flip(grid_scores,0), cmap=cm.coolwarm, alpha=.8)
    #plt.imshow(grid_scores, interpolation='bicubic',cmap=cm.coolwarm,extent=[minx,maxx,miny,maxy],aspect='auto')

    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()
    plt.set_xticks(())
    plt.set_yticks(())
    #fig.subplots_adjust(hspace=0.001,wspace =0.001)


    return

# ----------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_descriptions(plt,figure,filename_scores_pos, filename_scores_neg,delim=' ',caption='',inverse_score=0):

    scores_pos = []
    scores_neg = []
    files_pos = []
    files_neg = []

    with open(filename_scores_pos) as f:
        lines = f.read().splitlines()
    for each in lines:
        filename = each.split(delim)[0]
        value = each.split(delim)[1]
        if ((value[0]=='+')or(value[0]=='-')):
            if ((value[1]>='0') and (value[1]<='9')):
                scores_pos.append(value)
                files_pos.append(filename)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores_pos.append(value)
                files_pos.append(filename)


    with open(filename_scores_neg) as f:
        lines = f.read().splitlines()
    for each in lines:
        filename = each.split(delim)[0]
        value = each.split(delim)[1]
        if ((value[0]=='+')or(value[0]=='-')):
            if ((value[1]>='0') and (value[1]<='9')):
                scores_neg.append(value)
                files_neg.append(filename)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores_neg.append(value)
                files_neg.append(filename)

    scores_pos = numpy.array(scores_pos)
    scores_neg = numpy.array(scores_neg)

    for i in range(0,scores_pos.shape[0]):
        scores_pos[i]=scores_pos[i].split('x')[0]

    for i in range(0,scores_neg.shape[0]):
        scores_neg[i]=scores_neg[i].split('x')[0]

    scores_pos = scores_pos.astype(numpy.float32)
    scores_neg = scores_neg.astype(numpy.float32)

    if(inverse_score==1):
        scores_neg[:] = 1.0 -scores_neg[:]

    labels = numpy.hstack((numpy.full(len(scores_pos), 1), numpy.full(len(scores_neg), 0)))
    scores = numpy.hstack((scores_pos, scores_neg))




    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)



    if(roc_auc>0.5):
        plot_tp_fp(plt,figure,tpr, fpr,roc_auc,caption)
    else:
        plot_tp_fp(plt, figure, fpr, tpr, 1-roc_auc,caption)


# ----------------------------------------------------------------------------------------------------------------------
def display_distributions(plt,fig,path_scores1, path_scores2,delim=' ',inverse_score=0):
    scores1 = []
    scores2 = []


    with open(path_scores1) as f:
        lines = f.read().splitlines()
    for each in lines:
        value = each.split(delim)[1]
        if ((value[0] == '+') or (value[0] == '-')):
            if ((value[1] >= '0') and (value[1] <= '9')):
                scores1.append(value)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores1.append(value)


    with open(path_scores2) as f:
        lines = f.read().splitlines()
    for each in lines:
        value = each.split(delim)[1]
        if ((value[0] == '+') or (value[0] == '-')):
            if ((value[1] >= '0') and (value[1] <= '9')):
                scores2.append(value)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores2.append(value)

    scores1 = numpy.array(scores1)
    scores2 = numpy.array(scores2)

    for i in range(0,scores1.shape[0]):
        scores1[i] = scores1[i].split('x')[0]

    for i in range(0,scores2.shape[0]):
        scores2[i] = scores2[i].split('x')[0]


    if(numpy.max(numpy.array(scores1).astype(float))<=1):
        scores1=100*numpy.array(scores1).astype(float)

    if (numpy.max(numpy.array(scores2).astype(float)) <= 1):
        scores2 = 100 * numpy.array(scores2).astype(float)

    if (inverse_score == 1):
        scores2 = 100-scores2

    scores1= scores1.astype(numpy.float32)
    scores2= scores2.astype(numpy.float32)

    m1 = numpy.min(scores1)
    m2 = numpy.min(scores2)
    min = numpy.minimum(m1,m2)
    scores1+= -min
    scores2+= -min


    freq1=numpy.bincount(numpy.array(scores1).astype(int))/len(scores1)
    freq2=numpy.bincount(numpy.array(scores2).astype(int))/len(scores2)


    x_max1=numpy.max(numpy.array(scores1).astype(float))
    x_max2=numpy.max(numpy.array(scores2).astype(float))
    y_max1=numpy.max(numpy.array(freq1).astype(float))
    y_max2=numpy.max(numpy.array(freq2).astype(float))

    x_max=max(x_max1,x_max1)*1.1
    y_max=max(y_max1,y_max2)*1.1

    #major_ticks = numpy.arange(0, x_max, 10)
    #minor_ticks = numpy.arange(0, x_max, 1)
    #plt.xlim([0.0, x_max])
    #plt.ylim([0.0, y_max])

    plt.grid(which='major',axis='both', color='lightgray',linestyle='--')
    #plt.minorticks_on()
    #plt.grid(which='minor', color='r')
    #plt.grid(which='both')


    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='on', labelleft='off', labeltop='off',labelright='off', labelbottom='off')

    plt.plot(freq1, color='red', lw=2)
    plt.plot(freq2, color='gray', lw=2)

    #plt.show()
#----------------------------------------------------------------------------------------------------------------------
def from_categorical(Y_2d):
    u = numpy.unique(Y_2d)
    Y = numpy.zeros(Y_2d.shape[0]).astype(int)
    for i in range(0, Y.shape[0]):
        #debug = Y_2d[i]
        index = smart_index(Y_2d[i],1)[0]
        Y[i]=index

    return Y
# ----------------------------------------------------------------------------------------------------------------------
def to_categorical(Y):
    u = numpy.unique(Y)
    Y_2d = numpy.zeros((Y.shape[0],u.shape[0])).astype(int)
    for i in range(0, Y.shape[0]):
        index = smart_index(u,Y[i])
        Y_2d[i,index]=1
    return Y_2d

# --------------------------------------------------------------------------------------------------------------------
