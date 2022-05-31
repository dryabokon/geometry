import cv2
import sys
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
from detector import detector_YOLO3
# ---------------------------------------------------------------------------------------------------------------------
filename_in = 'images/ex13/01.jpg'
filename_out = 'images/output/res.png'
folder_in =  './images/ex_bgrem/'
folder_out = './images/output/'
USE_CAMERA = True
# ---------------------------------------------------------------------------------------------------------------------
def demo_bg_removal_cam():
	if USE_CAMERA:
		cap = cv2.VideoCapture(0)
	else:
		cap = []
		frame = cv2.imread(filename_in)

	fgbg = cv2.createBackgroundSubtractorMOG2()

	while (True):
		if USE_CAMERA:
			ret, frame = cap.read()

		fgmask = fgbg.get_flow_image(frame)
		tools_image.desaturate_2d(frame)

		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray_rgb = tools_image.desaturate(frame)

		cv2.imshow('frame', fgmask)
		if cv2.waitKey(1) & 0xFF == 27:
			break

	if USE_CAMERA:
		cap.release()

	cv2.destroyAllWindows()

	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_bg_removal_viseo(folder_in,folder_out):
	filenames = tools_IO.get_filenames(folder_in,'*.jpg')
	#frame = cv2.imread(filename_in)
	fgbg = cv2.createBackgroundSubtractorMOG2()

	for filename in filenames:
		frame = cv2.imread(folder_in+filename)


		fgmask = fgbg.get_flow_image(frame)
		cv2.imwrite(folder_out+filename,fgmask)

		# frame = tools_image.desaturate_2d(frame)
		# tools_image.do_blend()
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray_rgb = tools_image.desaturate(frame)

	return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


	model_in = './data/ex_YOLO/models/model_default.h5'
	metadata_in = './data/ex_YOLO/models/metadata_default.txt'
	filename_image = './data/ex_detector/bike/Image.png'
	D = detector_YOLO3.detector_YOLO3(model_in, metadata_in)
	D.process_file(filename_image, './data/output/res_yolo.jpg')
