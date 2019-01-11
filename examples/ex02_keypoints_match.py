import cv2
import os
import numpy
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
import tools_alg_match
import tools_image
import tools_draw_numpy
import tools_calibrate
import tools_IO


# ---------------------------------------------------------------------------------------------------------------------
def example_find_matches_for_homography(detector='SIFT', matchtype='knn'):
	folder_input = 'images/ex02/'
	img1 = cv2.imread(folder_input + 'left.jpg')
	img2 = cv2.imread(folder_input + 'rght.jpg')
	img1_gray_rgb = tools_image.desaturate(img1)
	img2_gray_rgb = tools_image.desaturate(img2)

	folder_output = 'images/output/'

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	#else:
	#	tools_IO.remove_files(folder_output)

	points1, des1 = tools_alg_match.get_keypoints_desc(img1, detector)
	points2, des2 = tools_alg_match.get_keypoints_desc(img2, detector)

	match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2, matchtype)
	homography = tools_calibrate.get_homography_by_keypoints_desc(points1, des1, points2, des2, matchtype)

	if homography is None:
		return

	match2_cand = cv2.perspectiveTransform(match1.reshape(-1, 1, 2).astype(numpy.float32), homography)

	hit = numpy.zeros(match1.shape[0])

	for i in range(0, match1.shape[0]):
		if numpy.linalg.norm(match2_cand[i] - match2[i]) <= 5:
			hit[i] = 1
			img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(match1[i][1]), int(match1[i][0]), 3,[32, 192, 0])
			img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(match2[i][1]), int(match2[i][0]), 3,[32, 192, 0])
		else:
			img1_gray_rgb = tools_draw_numpy.draw_circle(img1_gray_rgb, int(match1[i][1]), int(match1[i][0]), 3,[0, 64, 255])
			img2_gray_rgb = tools_draw_numpy.draw_circle(img2_gray_rgb, int(match2[i][1]), int(match2[i][0]), 3,[0, 64, 255])

	img2_gray_rgb, img1_gray_rgb = tools_calibrate.get_stitched_images_using_homography(img2_gray_rgb, img1_gray_rgb,homography)

	roc_auc = 0
	if hit.size >= 2:
		fpr, tpr, thresholds = metrics.roc_curve(hit, -distance)
		roc_auc = auc(fpr, tpr)
		tools_IO.plot_tp_fp(plt, plt.figure(), tpr, fpr, roc_auc)
		plt.savefig(folder_output + ('_ROC_%s_%s_auc_%1.3f.png' % (detector, matchtype, roc_auc)))

	cv2.imwrite(folder_output + ('%s_%s_auc_%1.3f_left_matches.png' % (detector, matchtype, roc_auc)), img1_gray_rgb)
	cv2.imwrite(folder_output + ('%s_%s_auc_%1.3f_rght_matches.png' % (detector, matchtype, roc_auc)), img2_gray_rgb)

	return


# --------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	example_find_matches_for_homography('ORB', 'knn')
	example_find_matches_for_homography('ORB', 'flann')
	example_find_matches_for_homography('ORB', 'xxx')

	example_find_matches_for_homography('SIFT', 'knn')
	example_find_matches_for_homography('SIFT', 'flann')
	example_find_matches_for_homography('SIFT', 'xxx')

	example_find_matches_for_homography('SURF', 'knn')
	example_find_matches_for_homography('SURF', 'flann')
	example_find_matches_for_homography('SURF', 'xxx')
