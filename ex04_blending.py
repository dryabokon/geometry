import cv2
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_calibrate
import tools_IO
import tools_alg_match
import tools_image


# ---------------------------------------------------------------------------------------------------------------------
def example_04_blend_white_avg():
	folder_input = 'images/ex_blend/'
	img1 = cv2.imread(folder_input + 'white_L.png')
	img2 = cv2.imread(folder_input + 'white_R.png')

	folder_output = 'images/output/'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	cv2.imwrite(folder_output + 'avg.png', tools_image.blend_avg(img1, img2, (255, 255, 255)))

	return


# ---------------------------------------------------------------------------------------------------------------------
def example_04_blend_black_avg():
	folder_input = 'images/ex_blend/'
	img1 = cv2.imread(folder_input + 'black_L.png')
	img2 = cv2.imread(folder_input + 'black_R.png')

	folder_output = 'images/output/'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	cv2.imwrite(folder_output + 'avg.png', tools_image.blend_avg(img1, img2, (0, 0, 0)))

	return


# ---------------------------------------------------------------------------------------------------------------------

def example_04_blend_multi_band_black():
	folder_input = 'images/ex_blend/'
	img1 = cv2.imread(folder_input + 'black_L.png')
	img2 = cv2.imread(folder_input + 'black_R.png')
	folder_output = 'images/output/'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	cv2.imwrite(folder_output + 'blend_multi_band.png', tools_image.blend_multi_band(img1, img2,(0, 0, 0)))
	return


# ---------------------------------------------------------------------------------------------------------------------

def example_04_blend_multi_band_white():
	folder_input = 'images/ex_blend/'
	img1 = cv2.imread(folder_input + 'white_L.png')
	img2 = cv2.imread(folder_input + 'white_R.png')
	folder_output = 'images/output/'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	cv2.imwrite(folder_output + 'blend_multi_band.png', tools_image.blend_multi_band(img1, img2,(255, 255, 255)))
	return


# ---------------------------------------------------------------------------------------------------------------------
def example_04_find_homography_blend_multi_band():

	folder_input = 'images/ex_keypoints/'
	img_left = cv2.imread(folder_input + 'left.jpg')
	img_right = cv2.imread(folder_input + 'rght.jpg')

	folder_output = 'images/output/'
	output_left = folder_output + 'left_out.jpg'
	output_rght = folder_output + 'rght_out.jpg'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	points_left, des_left = tools_alg_match.get_keypoints_desc(img_left)
	points_rght, des_rght = tools_alg_match.get_keypoints_desc(img_right)

	homography = tools_calibrate.get_homography_by_keypoints_desc(points_left, des_left, points_rght, des_rght)
	result_left, result_right= tools_calibrate.get_stitched_images_using_homography(img_right, img_left, homography)
	result_image = tools_image.blend_multi_band(result_left,result_right)
	cv2.imwrite(output_left, result_image)

	homography = tools_calibrate.get_homography_by_keypoints_desc(points_rght, des_rght, points_left, des_left)
	result_right, result_left= tools_calibrate.get_stitched_images_using_homography(img_left, img_right, homography)
	result_image = tools_image.blend_multi_band(result_left,result_right)
	cv2.imwrite(output_rght, result_image)
	return


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	#example_04_blend_black_avg()
	#example_04_blend_white_avg()
	#example_04_blend_multi_band_black()
	#example_04_blend_multi_band_white()
	example_04_find_homography_blend_multi_band()

