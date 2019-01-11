import numpy
from skimage.draw import circle, line_aa


# ----------------------------------------------------------------------------------------------------------------------
def draw_circle(array_rgb, row, col, rad, color, alpha_transp=0):
    color = numpy.array(color)
    res_rgb = array_rgb.copy()
    if alpha_transp > 0:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_rgb.shape)] = array_rgb[circle(int(row), int(col), int(rad), shape=array_rgb.shape)]* alpha_transp + color * (1 - alpha_transp)
    else:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_rgb.shape)] = color
    return res_rgb


# ----------------------------------------------------------------------------------------------------------------------
def draw_line(array_rgb, row1, col1, row2, col2, color, alpha_transp=0):
    res_rgb = array_rgb.copy()

    rr, cc, vv = line_aa(int(row1), int(col1), int(row2), int(col2))
    for i in range(0, rr.shape[0]):
        if (rr[i] >= 0 and rr[i] < array_rgb.shape[0] and cc[i] >= 0 and cc[i] < array_rgb.shape[1]):
            clr = numpy.array(color) * vv[i] + array_rgb[rr[i], cc[i]] * (1 - vv[i])
            if alpha_transp > 0:
                res_rgb[rr[i], cc[i]] = clr * (1 - alpha_transp) + array_rgb[rr[i], cc[i]] * alpha_transp
            else:
                res_rgb[rr[i], cc[i]] = clr
    return res_rgb


# ----------------------------------------------------------------------------------------------------------------------
def draw_rect(array_rgb, row1, col1, row2, col2, color, alpha_transp=0):
    res_rgb = array_rgb.copy()
    res_rgb = draw_line(res_rgb, row1, col1, row1, col2, color, alpha_transp)
    res_rgb = draw_line(res_rgb, row1, col2, row2, col2, color, alpha_transp)
    res_rgb = draw_line(res_rgb, row2, col2, row2, col1, color, alpha_transp)
    res_rgb = draw_line(res_rgb, row2, col1, row1, col1, color, alpha_transp)
    return res_rgb

# ----------------------------------------------------------------------------------------------------------------------
