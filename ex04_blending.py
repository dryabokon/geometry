import cv2
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
from CV import tools_calibrate
from CV import tools_alg_match
# ---------------------------------------------------------------------------------------------------------------------
folder_output = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def example_01_blend_avg(img1,img2,background_color=(255, 255, 255)):
    cv2.imwrite(folder_output + 'avg.png', tools_image.blend_avg(img1, img2, background_color=background_color))
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_02_blend_multi_band(img1,img2,background_color=(255, 255, 255)):
    R = 500
    cv2.imwrite(folder_output + 'blend_multi_band.png', tools_image.blend_multi_band(img1, img2,background_color=background_color))
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_03_blend_multi_band_mask(img1,img2,background_color=(0, 0, 0)):
    adjust_colors = 'large'# small large
    R = 500
    res = tools_image.blend_multi_band_large_small0(img1, img2, background_color=background_color,filter_size=R, adjust_colors=adjust_colors, do_debug=True)
    cv2.imwrite(folder_output + 'blend_multi_band_mask_%s_R%02d.png'%(adjust_colors,R), res)
    return
# ---------------------------------------------------------------------------------------------------------------------
def example_04_find_homography_blend_multi_band(img_left,img_right,detector='SIFT', matchtype='knn'):

    output_left = folder_output + 'left_out.jpg'
    output_rght = folder_output + 'rght_out.jpg'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    points_left, des_left = tools_alg_match.get_keypoints_desc(img_left,detector)
    points_rght, des_rght = tools_alg_match.get_keypoints_desc(img_right,detector)


    homography1 = tools_calibrate.get_homography_by_keypoints_desc(points_left, des_left, points_rght, des_rght,matchtype=matchtype)
    result_left, result_right= tools_calibrate.get_stitched_images_using_homography(img_left, img_right, homography1)
    result_image = tools_image.blend_multi_band(result_left,result_right)
    cv2.imwrite(output_left, result_image)

    homography2 = tools_calibrate.get_homography_by_keypoints_desc(points_rght, des_rght, points_left, des_left,matchtype=matchtype)
    result_right, result_left= tools_calibrate.get_stitched_images_using_homography(img_right, img_left, homography2)

    result_image = tools_image.blend_multi_band(result_left,result_right)
    cv2.imwrite(output_rght, result_image)
    return
# ---------------------------------------------------------------------------------------------------------------------
img3 = cv2.imread('./images/ex_blend/part1.png')
img4 = cv2.imread('./images/ex_blend/part2.png')
# ---------------------------------------------------------------------------------------------------------------------
def tutorial1():
    # white
    # example_01_blend_avg(cv2.imread('./images/ex_blend/white_L.png'),
    #                      cv2.imread('./images/ex_blend/white_R.png'),
    #                      background_color=(255, 255, 255))

    example_02_blend_multi_band(cv2.imread('./images/ex_blend/finger_left.png'),
                                cv2.imread('./images/ex_blend/finger_right2.png'),
                                background_color=(255, 255, 255))
    return
# ---------------------------------------------------------------------------------------------------------------------
def tutorial2():
    # black
    example_01_blend_avg(cv2.imread('./images/ex_blend/black_L.png'),
                         cv2.imread('./images/ex_blend/black_R.png'),
                         background_color=(0, 0, 0))

    example_02_blend_multi_band(cv2.imread('./images/ex_blend/black_L.png'),
                                cv2.imread('./images/ex_blend/black_R.png'),
                                background_color=(0, 0, 0))
    return
# ---------------------------------------------------------------------------------------------------------------------
def tutorial3():

    example_04_find_homography_blend_multi_band(
        cv2.imread('./images/ex_keypoints/left.jpg'),
        cv2.imread('./images/ex_keypoints/rght.jpg'))
    return
# ---------------------------------------------------------------------------------------------------------------------
def tutorial4():
    example_03_blend_multi_band_mask(cv2.imread('./images/ex_blend/finger_left.png'),
                                     cv2.imread('./images/ex_blend/finger_right2.png'),
                                     background_color=(255,255,255))

    # example_03_blend_multi_band_mask(cv2.imread('./images/ex_blend/black_L.png'),
    #                                  cv2.imread('./images/ex_blend/black_R.png'),
    #                                  background_color=(0, 0, 0))

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_output)

    tutorial4()
