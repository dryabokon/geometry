import cv2
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_alg_match
import tools_image
import tools_draw_numpy
import tools_IO


# ----------------------------------------------------------------------------------------------------------------------
def example_harris_corner_detection(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	for each in tools_alg_match.get_corners_Harris(img):
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
def example_shi_tomasi_corner_detector(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	for each in tools_alg_match.get_corners_Shi_Tomasi(img):
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
def example_SIFT(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	points, desc = tools_alg_match.get_keypoints_desc_SIFT(img)

	for each in points:
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
def example_SURF(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	points, desc = tools_alg_match.get_keypoints_desc_SURF(img)

	for each in points:
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
def example_Fast_Corner_Detector(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	points = tools_alg_match.get_corners_Fast(img)

	for each in points:
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
def example_STAR_Detector(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	points = tools_alg_match.get_keypoints_STAR(img)

	for each in points:
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)


# ----------------------------------------------------------------------------------------------------------------------
def example_ORB(filename_in, filename_out):
	img = cv2.imread(filename_in)
	gray_rgb = tools_image.desaturate(img)

	points, desc = tools_alg_match.get_keypoints_desc_ORB(img)

	for each in points:
		gray_rgb = tools_draw_numpy.draw_circle(gray_rgb, each[1], each[0], 2, [0, 0, 255])

	cv2.imwrite(filename_out, gray_rgb)
	return


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	path_in = '_images/ex01/'
	filename_in = path_in + 'image01.png'

	path_out = '_images/output/'

	if not os.path.exists(path_out):
		os.makedirs(path_out)
	else:
		tools_IO.remove_files(path_out)

	example_harris_corner_detection(filename_in, path_out + 'harris.png')
	example_shi_tomasi_corner_detector(filename_in, path_out + 'shi.png')
	example_Fast_Corner_Detector(filename_in, path_out + 'fast_corners.png')
	example_STAR_Detector(filename_in, path_out + 'star.png')
	example_SIFT(filename_in, path_out + 'sift.png')
	example_SURF(filename_in, path_out + 'surf.png')
	example_ORB(filename_in, path_out + 'orb.png')
