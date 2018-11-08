import numpy
from skimage.draw import circle,line_aa,line
# ----------------------------------------------------------------------------------------------------------------------
def draw_circle(array_RGB,row,col,rad,color,alpha_transp=0):
    res_RGB = array_RGB.copy()
    if alpha_transp>0 :
        res_RGB[circle(row, col, rad, shape=array_RGB.shape)] = color
    else:
        res_RGB[circle(row, col, rad, shape=array_RGB.shape)] = color*(1-alpha_transp)+array_RGB[row,col]*alpha_transp
    return res_RGB
# ----------------------------------------------------------------------------------------------------------------------
def draw_line(array_RGB,row1,col1,row2,col2,color,alpha_transp=0):
    res_RGB=array_RGB.copy()

    rr,cc,vv = line_aa(row1, col1, row2, col2)
    for i in range(0,rr.shape[0]):
        if (rr[i]>=0 and rr[i]<array_RGB.shape[0] and cc[i]>=0 and cc[i]<array_RGB.shape[1] ):
            clr = numpy.array(color) * vv[i]  + array_RGB[rr[i],cc[i]]*(1-vv[i])
            if alpha_transp>0 :
                res_RGB[rr[i],cc[i]] = clr*(1-alpha_transp)+array_RGB[rr[i],cc[i]]*alpha_transp
            else:
                res_RGB[rr[i], cc[i]] = clr
    return res_RGB
# ----------------------------------------------------------------------------------------------------------------------
def draw_rect(array_RGB,row1,col1,row2,col2,color,alpha_transp=0):
    res_RGB = array_RGB.copy()
    res_RGB=draw_line(res_RGB, row1, col1  , row1  , col2, color,alpha_transp)
    res_RGB=draw_line(res_RGB, row1, col2  , row2  , col2, color,alpha_transp)
    res_RGB=draw_line(res_RGB, row2, col2  , row2  , col1, color,alpha_transp)
    res_RGB=draw_line(res_RGB, row2, col1  , row1  , col1, color,alpha_transp)
    return res_RGB
# ----------------------------------------------------------------------------------------------------------------------
