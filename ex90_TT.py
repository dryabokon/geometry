# ---------------------------------------------------------------------------------------------------------------------
def sort_KV(ids,values):

    t_list = [[i,v] for i,v in zip(ids,values)]
    t_list2 = sorted(t_list, key=lambda l: l[0],reverse=False)
    res = [v[1] for v in t_list2]

    return res
# ---------------------------------------------------------------------------------------------------------------------
def get_unique(A):
    dct = {}
    for a in A:
        dct[a]=0
    res = [a for a in dct.keys()]
    return res
# ---------------------------------------------------------------------------------------------------------------------
def get_format(I):
    i = I
    n = 1
    while float(i) / 10.0 >= 1:
        i = i / 10
        n += 1
    if n==1:
        res = '%d'
    else:
        res = '%0' + '%d' % n + 'd'
    return res
# ---------------------------------------------------------------------------------------------------------------------
str_input = 'Warsaw.jpg,2021-07-03 16:21:12.357246\n' \
            'Rome.png,2021-07-04 16:21:12.357246\n' \
            'Warsaw.png,2021-07-05 16:21:12.357246\n' \
            'London.png,2021-07-06 16:21:12.357246\n' \
            'Rome.png,2021-07-02 16:21:12.357246\n'\
            'Warsaw.jpg,2021-07-05 16:21:12.357246\n' \
            'Warsaw.png,2021-07-05 16:21:12.357246\n' \
            'Warsaw.png,2021-07-05 16:21:12.357246\n' \
            'Warsaw.bmp,2021-07-25 16:21:12.357246\n' \
            'Warsaw.png,2021-07-26 16:21:12.357246\n' \
            'Warsaw.png,2021-07-27 16:21:12.357246\n' \
            'Warsaw.png,2021-07-28 16:21:12.357246\n' \
            'Warsaw.png,2021-07-02 16:21:12.357246' \
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #parse input string to a list of rows
    rows = str_input.split('\n')
    names,exts,times,idx = [],[],[],[]
    for i,row in enumerate(rows):
        items = row.split(',')
        names.append(items[0].split('.')[0])
        exts.append(items[0].split('.')[-1])
        times.append(items[1])
        idx.append(i)

    #iterate over unique names, fetch timestamp rank for groups of unique filenames
    res_idx,res_name,res_suffix,res_ext = [],[],[],[]
    for uname in get_unique(names):
        I,T,E = [],[],[]
        for name,ext,time,i in zip(names,exts,times,idx):
            if name==uname:
                I.append(i)
                T.append(time)
                E.append(ext)

        res_idx+=sort_KV(T,I)
        res_ext+=sort_KV(T,E)
        res_suffix+=[get_format(len(I))%(1+i) for i in range(len(I))]
        res_name+=([uname]*len(I))

    #re-order list of records according to original index of filenames
    res_name = sort_KV(res_idx, res_name)
    res_ext = sort_KV(res_idx, res_ext)
    res_suffix = sort_KV(res_idx,res_suffix)

    #Construct output string by joining the items in the list
    res = [name+suffix+'.'+ext for name,suffix,ext in zip(res_name,res_suffix,res_ext)]

    print('\n'.join(res))
