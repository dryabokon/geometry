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
# ----------------------------------------------------------------------------------------------------------------------
def smart_index(array, value):
    return numpy.array([i for i, v in enumerate(array) if (v == value)])
# ----------------------------------------------------------------------------------------------------------------------
def create_folders(base_folder):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    else:
        remove_files(base_folder)
        # remove_folders(base_folder)

    for each in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "@", "B", "C", "D", "E", "F", "G", "H", "I",
                 "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
        if not os.path.exists(base_folder + each + '/'):
            os.makedirs(base_folder + each + '/')
        else:
            remove_files(base_folder + each + '/')

    return
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
def save_images(path, images, labels=None, filenames=None,descriptions=None):

    if descriptions is not None:
        f_handle = open(path+"descript.ion", "a+")

    for i in range(0,len(images)):
        if filenames is not None:
            short_name = filenames[i]
        else:
            short_name = "%s_%05d.bmp" % (labels[i], i)

        images[i].save(path + short_name, "bmp")

        if descriptions is not None:
            f_handle.write("%s %s\n" % (short_name, descriptions[i]))

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
def load_mapping(file_mapping,delim='\t'):
    m = load_mat(file_mapping, dtype=numpy.str, delim=delim)
    mapping = {'key': 'value'}
    for i in range (0,m.shape[0]):
        mapping.update({m[i,0]: m[i,1]})

    del mapping['key']

    return mapping
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
def load_mat_by_lines(filename,lines,dtype=numpy.int,delim='\t'):
    f = open(filename, 'rb')

    mat = []
    for i, line in enumerate(f):
        if i in lines:
            vec = line.decode().split('\t')
            mat.append(vec)

    mat = numpy.array(mat)
    f.close()

    return mat
# ----------------------------------------------------------------------------------------------------------------------
def load_mat(filename,dtype=numpy.int,delim='\t',lines=None):
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
def resize_tensor(X, new_width, new_height):
    resized_X=[]
    for i in range (0,X.shape[0]):

        array = X[i]
        if (len(X[i].shape)==3) and (X[i].shape[2]==1):
            array = numpy.reshape(array,(array.shape[0],array.shape[1]))

        image = (toimage(array)).resize((new_width, new_height))
        image = image.convert("RGB")
        resized_array = (numpy.array(image)).reshape((new_width, new_height, 3))

        resized_X.append(resized_array)

    return numpy.array(resized_X)
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
'''def resize_image(image,cols,rows,mode='scale'):
    result = image.resize((resize_W, resize_H))

    w,h = image.size
    ratio = w/h

    if mode == 'fit_height':
        result = image.resize((resize_H*ratio, resize_H))

        if(resize_H*ratio - resize_W>0):
            (left, upper, right, lower) =   (int((resize_H*ratio - resize_W)/2), 0, int(resize_W-(resize_H*ratio - resize_W)/2), resize_H)
            result result.crop((left, upper, right, lower))

    return result
'''
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
def load_all_aligned_images(path, mask="*.bmp", patterns=numpy.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),index=None,exclusion_folder=None, limit=None, resize_W=None, resize_H=None):
    print("Loading images from ", path)


    images = []
    labels = []
    filenames = []
    DX = 0
    DY = 0

    for l in range(0,patterns.shape[0]):
        print(patterns[l], end=' ')
        if l == 0:
            (images,  labels,  filenames,  DX, DY) = load_aligned_images_from_folder(path + (patterns[l] + "/"), patterns[l], mask,exclusion_folder, limit, resize_W,resize_H)
            if(index!=None):
                images = images[index[l]]
                labels = labels[index[l]]
                filenames = filenames[index[l]]
        else:
            (timages, tlabels, tfilenames, DX, DY) = load_aligned_images_from_folder(path + (patterns[l] + "/"), patterns[l], mask,exclusion_folder, limit, resize_W,resize_H)
            if(index!=None):
                timages = timages[index[l]]
                tlabels = tlabels[index[l]]
                tfilenames = tfilenames[index[l]]
            images = numpy.vstack((images, timages))
            labels = numpy.hstack((labels, tlabels))
            filenames = numpy.hstack((filenames, tfilenames))

    print()
    return (images, labels, filenames, DX, DY)
# ----------------------------------------------------------------------------------------------------------------------
def transform_flat_images(flat_array, rows, cols):
    res = []

    for i in range(0, flat_array.shape[0]):
        img = toimage(numpy.array(flat_array[i, :]).reshape(cols, rows))
        img = img.resize((83, 100))

        resized = toimage(numpy.zeros((100, 120)))
        pos = (120 - img.width) / 2
        pos = int(pos)
        resized.paste(img, ((pos, 0)))
        for col in range(0, resized.width):
            resized.putpixel((col, 0), 0)

        # for row in range(0, resized.height):
        #    for col in range (0,resized.width):
        #        resized.putpixel((col, 0), 0)

        res.append(numpy.array(resized).flatten())

    return numpy.array(res)


# ----------------------------------------------------------------------------------------------------------------------
def save_fails(path, DX, DY, Xtest, ytest, ypred, filenames=None, remove_flag=1):

    if path == None:
        return

    if remove_flag == 1:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            remove_files(path)

    if os.path.exists(path + "descript.ion"):
        f_handle = open(path + "descript.ion", "a")
    else:
        f_handle = open(path + "descript.ion", "w")

    j = 0
    for i in range(0, ytest.shape[0]):
        if ytest[i] != ypred[i]:
            if filenames is None:
                localname = "%d_%05d.bmp" % (ypred[i], j)
            else:
                localname = filenames[i][:len(filenames[i]) - 4] + ("_%s" % ytest[i]) + ".bmp"

            toimage(Xtest[i].reshape(DY, DX)).save(path + localname, "bmp")
            f_handle.write("%s %s\n" % (localname, ypred[i]))
            j = j + 1

    f_handle.close()

    return


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
def copy_fails(path_images,names_on_disk,labels_fact, labels_pred,patterns,path_output):

    f_handle = open(path_output + "descript.ion", "w")

    for i in range (0,labels_fact.shape[0]):
        if(labels_fact[i]!=labels_pred[i]):
            dest_name = names_on_disk[i].rsplit('/', 1)[1]
            dest_name = ('%s_' % patterns[int(labels_fact[i])]) + dest_name
            copyfile(path_images+names_on_disk[i], path_output+dest_name)
            f_handle.write("%s %s\n" % (dest_name, patterns[int(labels_pred[i])]))

    f_handle.close()

    return
# ----------------------------------------------------------------------------------------------------------------------
def split_samples(input_folder, folder_part1, folder_part2, ratio=0.5):
    print("Split samples..")
    if not os.path.exists(folder_part1):
        os.makedirs(folder_part1) #096 109 93 44
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


# ----------------------------------------------------------------------------------------------------------------------
def resize_samples(input_folder, output_folder, rows, cols):
    print("Resize samples..")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        remove_files(output_folder)
        remove_folders(output_folder)

    folder_list = [f for f in os.listdir(input_folder)]

    for f in folder_list:
        folder_name = input_folder + f
        if os.path.isdir(folder_name):
            os.makedirs(folder_part1 + f)
            os.makedirs(folder_part2 + f)
            print(f)
            file_list = [f for f in os.listdir(folder_name)]
            for file_name in file_list:
                img = Image.open(input_folder + file_name)
                img.thumbnail((rows, cols), Image.ANTIALIAS)
                img.save(output_folder + file_name, "bmp")
    return
# ----------------------------------------------------------------------------------------------------------------------
def binarize_and_normalize_images(input_folder, output_folder, prefix, norm_size_rows, norm_size_cols):
    if not os.path.exists(input_folder):
        return

    print("Bin and norm images from " + input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        remove_files(output_folder)
        remove_folders(output_folder)

    file_list = [f for f in os.listdir(input_folder)]

    i = 0
    for f in file_list:
        file_name = input_folder + f
        img = numpy.array(Image.open(file_name).convert('L'))
        for level in set((0.95, 1.00, 1.05)):
            res_image = bin_by_global_white_level(img, 5, level)
            res_image[0, :] = 255
            res_image[:, 0] = 255
            res_image[res_image.shape[0] - 1, :] = 255
            res_image[:, res_image.shape[1] - 1] = 255

            mask1 = mask.Mask()
            mask1.init_from_numpy_array(res_image)
            mask1.filter_noise(2, 2)
            mask1.filter_leave_the_largest_contour()
            mask1.__calc_contour_sizes__()

            if (mask1.X.shape[0] > 5) and (mask1.ConS.shape[0] > 0) and (
                        mask1.ConS.max() > 0.25 * res_image.shape[0] * res_image.shape[1]):
                res_image = mask1.resotre_to_array()
                res_image = numpy.array(toimage(res_image).convert('L'))
                res_image = align_to_sizes(res_image, norm_size_cols,norm_size_rows)
                toimage(res_image).save(output_folder + prefix + ("_%06d.bmp" % i), "bmp")
                i += 1

    return


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def binarize_and_normalize_all_images(input_folder, output_folder, norm_size_rows, norm_size_cols,
                                      patterns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        remove_files(output_folder)
        remove_folders(output_folder)

    for l in patterns:
        prefix = l
        binarize_and_normalize_images(input_folder + l + "/", output_folder + l + "/", prefix, norm_size_rows,norm_size_cols)
    return


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sort_labels(out_filename,delim='\t'):

    mat1 = load_mat(out_filename, dtype=numpy.chararray, delim=delim)
    hdr1 = mat1[0, :]
    mat1 = mat1[1:, :]
    mat1 = numpy.vstack((hdr1, mat1[mat1[:, 0].argsort()])).astype(numpy.str)

    save_mat(mat1, out_filename, fmt='%s', delim=delim)

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
    sort_labels(out_filename,delim)

    return
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_ACDSee_description(path, filenames, labels_pred, labels_fact, challangers, challangers_prob,append=0):
    folders = os.listdir(path)

    f_handle = open(path + "descript.ion", "w")
    i = 0

    for each in filenames:
        idx = smart_index(challangers[i, :], labels_fact[i])[0]
        prb = challangers_prob[i, idx]

        if labels_pred[i] == labels_fact[i]:
            f_handle.write("%s %f\n" % (each, prb))
        else:
            f_handle.write("%s %fx\n" % (each, prb))
        i += 1

    f_handle.close()

    for each in folders:
        if not os.path.isfile(path + each):
            if os.path.exists(path + each + '/' + "descript.ion"):
                os.remove(path + each + '/' + "descript.ion")
            copyfile(path + "descript.ion", path + each + '/' + "descript.ion")

    remove_file(path + "descript.ion")

    return


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def align_to_sizes(original_matrix, SX, SY):
    # print(originalImage)

    if original_matrix[0, 0] > 0:
        background_color = 0
    else:
        background_color = original_matrix.max()

    top = 0
    flag = 0
    i = 0
    while (flag == 0) and (i < original_matrix.shape[0]):
        if background_color not in original_matrix[i, :]:
            top += 1
        else:
            flag = 1
        i += 1

    i = original_matrix.shape[0] - 1
    bottom = i
    flag = 0
    while (flag == 0) and (i > 0):
        if background_color not in original_matrix[i, :]:
            bottom -= 1
        else:
            flag = 1
        i -= 1

    left = 0
    flag = 0
    j = 0
    while (flag == 0) and (j < original_matrix.shape[1]):
        if background_color not in original_matrix[:, j]:
            left += 1
        else:
            flag = 1
        j += 1

    j = original_matrix.shape[1] - 1
    right = j
    flag = 0
    while (flag == 0) and (j > 0):
        if background_color not in original_matrix[:, j]:
            right -= 1
        else:
            flag = 1
        j -= 1

    resImage = numpy.zeros((SY, SX))
    resImage[:] = 0

    scale = (float)(bottom + 1 - top) / (SY)
    iShift = (float)(left + right) / 2 - left - (SX / 2) * scale

    for row in range(1, SY - 1):
        for col in range(0, SX):
            r = int(top + (float)(row) * scale)
            c = int(left + (float)(col) * scale + iShift)
            r = max(0, min(r, original_matrix.shape[0] - 1))
            c = max(0, min(c, original_matrix.shape[1] - 1))
            resImage[row, col] = original_matrix[r, c]

    return resImage


# ----------------------------------------------------------------------------------------------------------------------
def bin_by_global_white_level(input_image, block_size, fWhites):
    th_image = threshold_otsu(input_image)

    return numpy.array(255 * (input_image > float(fWhites) * th_image )).astype(int)


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
def extract_LP_rect_from_jpeg(filename):
    f_handle = open(filename, "rb")

    f_handle.seek(0, 2)
    siz = f_handle.tell()
    f_handle.seek(f_handle.tell() - 650, 0)

    bufer = []
    key = f_handle.read(1)
    while (key != b''):
        key = f_handle.read(1)
        key_hex = key.hex()
        if (key_hex >= '10' and key_hex < '80'):
            x = bytearray.fromhex(key_hex).decode()
            bufer.append(x)

    f_handle.close()

    bufer = numpy.array(list(bufer)).astype(numpy.chararray)
    bufer = ''.join(bufer)

    coords = []

    frame_text = bufer.split("Frame=")[1].split(";")[0]
    coord_text = frame_text.split(',')
    for each in coord_text:
        t = re.sub('[^A-Za-z0-9]+', '', each)
        coords.append(t)

    coords = numpy.array(coords)
    if (coords.shape[0] == 8):
        return coords.astype(int)
    else:
        return 0


# ----------------------------------------------------------------------------------------------------------------------
def list_to_chararray(input_list):
    bufer = numpy.array(list(input_list)).astype(numpy.chararray)
    bufer = ''.join(bufer)
    return bufer


# ----------------------------------------------------------------------------------------------------------------------
def extract_rects_from_jpeg(filename, col_left, row_top):
    f_handle = open(filename, "rb")

    f_handle.seek(0, 2)
    siz = f_handle.tell()
    f_handle.seek(f_handle.tell() - 650, 0)

    bufer = []
    key = f_handle.read(1)
    while (key != b''):
        key = f_handle.read(1)
        key_hex = key.hex()
        if (key_hex >= '10' and key_hex < '80'):
            x = bytearray.fromhex(key_hex).decode()
            bufer.append(x)

    f_handle.close()

    bufer = numpy.array(list(bufer)).astype(numpy.chararray)
    bufer = ''.join(bufer)

    coords = []
    LP_text = []

    tmp = bufer.split("_CH0=")
    if (len(tmp) > 1):
        tmp = tmp[1]

        LP_text = bufer.split("_CH0=")[1].split(';')
        for each in LP_text:
            t = each.split('=')
            if (len(t) == 1):
                t = t[0]
            else:
                t = t[1]

            s = t.split(',')

            if (len(s) == 4):
                coords.append(s)

        LP_text = bufer.split("LicensePlate=")[1].split(';')[0]
        LP_text = re.sub('[^A-Za-z0-9]+', '', LP_text)

    coords = numpy.array(coords).astype(int)

    for row in range(0, coords.shape[0]):
        coords[row, 2] += coords[row, 0]
        coords[row, 3] += coords[row, 1]
        coords[row, 0] -= col_left
        coords[row, 2] -= col_left
        coords[row, 1] -= row_top
        coords[row, 3] -= row_top

    if (len(LP_text) != coords.shape[0]):
        coords = []
        LP_text = []

    coords = numpy.array(coords).astype(int)

    class Real(object):
        pass

    r = Real()
    setattr(r, "labels", LP_text)
    setattr(r, "rects", coords)

    return r


# ----------------------------------------------------------------------------------------------------------------------
def calc_hits_Agarkov(tol, pred_s_rect, real_s_rect):
    accuracy = 0
    best_l = numpy.full(real_s_rect.shape[0], -1).astype(int)
    all_l = numpy.zeros((real_s_rect.shape[0], pred_s_rect.shape[0])).astype(int)
    all_l[:] = -1

    if (pred_s_rect.shape[0] == 0):
        return int(accuracy), best_l, all_l

    for i in range(0, real_s_rect.shape[0]):
        minl = -1
        max_d = 0
        sq_real = (real_s_rect[i, 2] - real_s_rect[i, 0]) * (real_s_rect[i, 3] - real_s_rect[i, 1])
        for l in range(0, pred_s_rect.shape[0]):
            sq_pred = (pred_s_rect[l, 2] - pred_s_rect[l, 0]) * (pred_s_rect[l, 3] - pred_s_rect[l, 1])
            d = (float)(2 * intersection(pred_s_rect[l, 0], pred_s_rect[l, 2], pred_s_rect[l, 1], pred_s_rect[l, 3],
                                         real_s_rect[i, 0], real_s_rect[i, 2], real_s_rect[i, 1], real_s_rect[i, 3]) / (
                            sq_real + sq_pred))
            if (d >= tol):
                all_l[i, l] = l

            if ((d > max_d) and (d > 0)):
                minl = l
                max_d = d

        if (minl >= 0) and (max_d >= tol):
            best_l[i] = minl
            accuracy += 1

    return int(accuracy), best_l, all_l


# ----------------------------------------------------------------------------------------------------------------------
def write_candidates(base_folder, real_s, real_s_rect, pred_s_rect, candidates, cand_count):
    res_cand_count = cand_count
    (accuracy, hits, all_hits) = calc_hits_Agarkov(0.80, real_s_rect, pred_s_rect)

    for i in range(0, pred_s_rect.shape[0]):
        h = hits[i]
        if (h >= 0):
            img = candidates[i].reshape(120, 100)
            symb = real_s[h]
            idx = smart_index(
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                 "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"], symb)
            toimage(img).save(base_folder + symb + '/' + symb + '_' + ("%06d" % res_cand_count[idx]) + '.bmp', "bmp")
            res_cand_count[idx] += 1
        i += 1

    return res_cand_count


# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_Samples(filename,caption='',add_noice=0):

    data = load_mat(filename, numpy.chararray, '\t')
    labels = (data[:, 0]).astype('float32')
    X = data[:, 1:].astype('float32')

    X1 = X[labels >  0]
    X0 = X[labels <= 0]

    fig = plt.figure()
    fig.canvas.set_window_title(caption)

    ax = fig.gca()


    if add_noice>0:
        noice1 = 0.05-0.1*numpy.random.random_sample(X1.shape)
        noice0 = 0.05-0.1*numpy.random.random_sample(X0.shape)
        X1+=noice1
        X0+=noice0

    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()
    #fig.tight_layout(pad=0)

    #plt.show()
# ----------------------------------------------------------------------------------------------------------------------
def plot_tp_fp(plt,fig,tpr,fpr,roc_auc,caption=''):

    ax = fig.gca()
    ax.set_xticks(numpy.arange(0, 1.1, 0.1))
    ax.set_yticks(numpy.arange(0, 1.1, 0.1))

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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid()

    colors_list = list(('lightgray','red','blue','purple','green','orange','cyan'))
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
def get_roc_data_from_scores_file(path_scores,caption=''):

    data = load_mat(path_scores, numpy.chararray, ' ')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return tpr,fpr,roc_auc
# ----------------------------------------------------------------------------------------------------------------------
def display_roc_curve_from_file(plt,path_scores,caption=''):

    data = load_mat(path_scores, numpy.chararray, ' ')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plot_tp_fp(plt,tpr,fpr,roc_auc,caption)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_scores(plt,filename_data,filename_data_grid,filename_scores_grid,th,noice_needed=0,caption=''):

    data = load_mat(filename_scores_grid, numpy.chararray, ' ')
    grid_scores = data[:, 1:].astype('float32')

    data = load_mat(filename_data_grid, numpy.chararray, '\t')
    data_grid = data[:,1:].astype('float32')

    data = load_mat(filename_data, numpy.chararray, '\t')
    labels = (data[:, 0]).astype('float32')
    X = data[:,1:].astype('float32')
    X1 = X[labels >  0]
    X0 = X[labels <= 0]

    max = numpy.max(grid_scores)
    min = numpy.min(grid_scores)

    for i in range(0,grid_scores.shape[0]):
        if(grid_scores[i]>th):
            grid_scores[i]=(grid_scores[i]-th)/(max-th)
        else:
            grid_scores[i] = (grid_scores[i] - th) / (th-min)


    S=int(math.sqrt(grid_scores.shape[0]))
    grid_scores=numpy.reshape(grid_scores,(S,S))

    minx=numpy.min(data_grid[:, 0])
    maxx=numpy.max(data_grid[:, 0])
    miny=numpy.min(data_grid[:, 1])
    maxy=numpy.max(data_grid[:, 1])

    dx=(maxx-minx)
    dy=(maxy-miny)
    minx-=dx*0.05
    maxx+=dx*0.05
    miny-=dy*0.05
    maxy+=dy*0.05

    if noice_needed>0:
        noice1 = 0.05-0.2*numpy.random.random_sample(X1.shape)
        noice0 = 0.05-0.2*numpy.random.random_sample(X0.shape)
        X1+=noice1
        X0+=noice0



    plt.set_title(caption)


    plt.imshow(grid_scores, interpolation='bicubic',cmap=cm.coolwarm,extent=[minx,maxx,miny,maxy],aspect='auto')
    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()

    return
# ----------------------------------------------------------------------------------------------------------------------
def display_score_distribution_from_file(plt,path_scores,caption=''):


    data = load_mat(path_scores, numpy.chararray, ' ')
    labels = (data [:, 0]).astype('float32')
    scores = data[:, 1:].astype('float32')

    mn=numpy.min(scores)
    if mn<0:
        scores+= -mn

    scores1 = scores[labels > 0]
    scores2 = scores[labels <= 0]

    scores1 = numpy.array(scores1).astype(int)[:,0]
    scores2 = numpy.array(scores2).astype(int)[:,0]

    freq1=numpy.bincount(scores1)/len(scores1)
    freq2=numpy.bincount(scores2)/len(scores2)

    x_max1=numpy.max(numpy.array(scores1).astype(float))
    x_max2=numpy.max(numpy.array(scores2).astype(float))
    y_max1=numpy.max(numpy.array(freq1).astype(float))
    y_max2=numpy.max(numpy.array(freq2).astype(float))

    x_max=max(x_max1,x_max1)*1.1
    y_max=max(y_max1,y_max2)*1.1

    major_ticks = numpy.arange(0, x_max, 10)
    minor_ticks = numpy.arange(0, x_max, 1)


    plt.set_title(caption)

    #plt.xlim([0.0, x_max])
    #plt.ylim([0.0, y_max])

    plt.grid(which='major',axis='both', color='lightgray')
    plt.minorticks_on()

    plt.plot(freq1, color='red', lw=2,alpha = 0.4)
    plt.plot(freq2, color='blue', lw=2,alpha = 0.4)
    #plt.show()

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
        if ((value[0]=='+')or(value[0]=='+')):
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
        if ((value[0]=='+')or(value[0]=='+')):
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
        if ((value[0] == '+') or (value[0] == '+')):
            if ((value[1] >= '0') and (value[1] <= '9')):
                scores1.append(value)
        else:
            if ((value[0] >= '0') and (value[0] <= '9')):
                scores1.append(value)


    with open(path_scores2) as f:
        lines = f.read().splitlines()
    for each in lines:
        value = each.split(delim)[1]
        if ((value[0] == '+') or (value[0] == '+')):
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
#----------------------------------------------------------------------------------------------------------------------
def reshape_data_2d_4d(X_train,Y_train,X_test,Y_test,img_rows,img_cols):

    X_train_4d = X_train.reshape((X_train.shape[0],img_rows,img_cols, 1)).astype(numpy.float32)
    X_test_4d =  X_test.reshape ((X_test.shape[0] ,img_rows,img_cols, 1)).astype(numpy.float32)

    #X_train_4d /= 255
    #X_test_4d /= 255

    #Y_train_2d = np_utils.to_categorical(Y_train, 2)
    Y_train_2d = numpy.zeros((Y_train.shape[0],2)).astype(int)
    for i in range(0,Y_train.shape[0]):
        if int(Y_train[i])!=0:
            Y_train_2d[i,1]=1
        else:
            Y_train_2d[i,0]=1


    #Y_test_2d = np_utils.to_categorical(Y_test, 2)
    Y_test_2d = numpy.zeros((Y_test.shape[0],2))
    for i in range(0,Y_test.shape[0]):
        if int(Y_test[i])!=0:
            Y_test_2d[i,1]=1
        else:
            Y_test_2d[i,0]=1


    return (X_train_4d,Y_train_2d,X_test_4d, Y_test_2d)
# --------------------------------------------------------------------------------------------------------------------
def reshape_data_4d_42(X_train_4d,Y_train_2d,X_test_4d, Y_test_2d):

    X_train = X_train_4d.reshape(X_train_4d.shape[0],X_train_4d.shape[1]*X_train_4d.shape[2])
    X_test = X_test_4d.reshape(X_test_4d.shape[0], X_test_4d.shape[1] * X_test_4d.shape[2])

    Y_train = numpy.zeros(X_train.shape[0])
    for i in range(0,Y_train_2d.shape[0]):
        if Y_train_2d[i,0]==1:
            Y_train[i]=1

    Y_test = numpy.zeros(X_test.shape[0])
    for i in range(0, Y_test_2d.shape[0]):
        if Y_test_2d[i, 0] == 1:
            Y_test[i] = 1

    return X_train, Y_train, X_test, Y_test
# --------------------------------------------------------------------------------------------------------------------
def get_padding_from_LP_rect(LP_rect):
    col_left = min(LP_rect[0], LP_rect[2], LP_rect[4], LP_rect[6]) - 10
    col_right = max(LP_rect[0], LP_rect[2], LP_rect[4], LP_rect[6]) + 10
    row_top = min(LP_rect[1], LP_rect[3], LP_rect[5], LP_rect[7]) - 10
    row_bottom = max(LP_rect[1], LP_rect[3], LP_rect[5], LP_rect[7]) + 10
    return (col_left, col_right, row_top, row_bottom)
# --------------------------------------------------------------------------------------------------------------------
def split_file(file,file_part1,file_part2):
    with open(file) as f:
        lines = f.readlines()

    f1 = open(file_part1, 'w')
    f2 = open(file_part2, 'w')

    for i in range(1,len(lines)):

        if random.random()>0.5:
            f1.write(lines[i])
        else:
            f2.write(lines[i])
    f1.close()
    f2.close()

    return
# --------------------------------------------------------------------------------------------------------------------
def merge_scores(filename1,filename2,filename_res,delim1='\t',delim2='\t',delim_res='\t'):

    mat1=load_mat(filename1,dtype=numpy.chararray,delim=delim1)
    mat2=load_mat(filename2,dtype=numpy.chararray,delim=delim2)

    hdr1 = mat1[0, :]
    hdr2 = mat2[0, :]

    mat1 = mat1[1:, :]
    mat2 = mat2[1:, :]

    mat1=numpy.vstack((hdr1,mat1[mat1[:,0].argsort()]))
    mat2=numpy.vstack((hdr2,mat2[mat2[:,0].argsort()]))

    if(mat1.shape[0]==mat2.shape[0]):
        mat = numpy.hstack((mat1,mat2[:,1::1])).astype(numpy.str)
        save_mat(mat, filename_res, fmt='%s', delim=delim_res)
        #mat_debug = numpy.hstack((mat1, mat2)).astype(numpy.str)



    #save_mat(mat_debug, "D:/xxx.txt", fmt='%s', delim='\t')
    #save_mat(mat1, "D:/mat1.txt", fmt='%s', delim='\t')
    #save_mat(mat2, "D:/mat2.txt", fmt='%s', delim='\t')

    return
# --------------------------------------------------------------------------------------------------------------------
'''
def save_mnist_digits():

    path = 'D:/Projects/Num/Aroi/Ex08/'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    patterns = numpy.unique(y_test)

    for key in patterns:
        output_path = path + ('%d/' % key)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            remove_files(output_path)

    for key in patterns:
        index = smart_index(y_train,key)
        full_names = []
        for i in range(0,index.shape[0]):
            name = ('%d_%03d.bmp' % (key,i))
            full_names.append(name)

        output_path = path + ('%d/' % key)
        #IO.save_images          (output_path, x_train[index],         labels=None, filenames=full_names, descriptions=None)
        save_arrays_as_images(output_path, x_train.shape[1], x_train.shape[2], x_train[index], labels=None, filenames=full_names, descriptions=None)

    return
'''
# --------------------------------------------------------------------------------------------------------------------
def generate_metadata(path_input,file_output):

    vec = numpy.array(['filename','path','camera','plate','plate_left','plate_top','plate_width','plate_height','dataset','maker','model','make_model','angle']).astype(numpy.str)
    save_raw_vec(vec, file_output,fmt='%s',delim=',')

    patterns = get_sub_folder_from_folder(path_input)
    for each in patterns:
        local_filenames = fnmatch.filter(listdir(each), '*.jpg') + fnmatch.filter(listdir(each), '*.jpeg') + fnmatch.filter(listdir(path_input), '*.JPG')
        for filename in local_filenames:
            cls = each.split('/')[-1]
            vec = numpy.array([filename,'path','camera','plate','plate_left','plate_top','plate_width','plate_height','train','maker','model',cls,'front'])
            save_raw_vec(vec, file_output, fmt='%s', delim=',')

    return
# --------------------------------------------------------------------------------------------------------------------
def save_images(images,path):
    remove_files(path)

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, len(images)):
        toimage(images[i]).save(path+'%03d.bmp' % i)
    return
# ----------------------------------------------------------------------------------------------------------------------
