import cv2
import sys
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
#from detector import detector_YOLO3
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
def demo_bg_removal_video(folder_in,folder_out):
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
def remove_bg(folder_in, list_of_masks='*.jpg', limit=255):

	filenames = tools_IO.get_filenames(folder_in, list_of_masks)[:limit]
	image = cv2.imread(folder_in + filenames[0])
	image_S = numpy.zeros((image.shape[0], image.shape[1], 3), dtype='long')
	image_C = numpy.full((image.shape[0], image.shape[1]),1e-4, dtype='float')


	for filename in filenames:
		image = cv2.imread(folder_in + filename)
		image_S += image

	im_mean = image_S/(len(filenames)*1.0)

	th = 50
	for filename in filenames:
		image = cv2.imread(folder_in + filename)
		mask1 = 1*(abs(im_mean[:,:,0] - image[:,:,0])<th)
		mask2 = 1*(abs(im_mean[:,:,1] - image[:,:,1])<th)
		mask3 = 1*(abs(im_mean[:,:,2] - image[:,:,2])<th)

		mask = ((mask1 + mask2 + mask3)==0).astype(bool)
		mask = ~mask

		image[mask] = 0
		image_S += image
		image_C += mask

	for c in [0, 1, 2]:
		image_S[:, :,c] = image_S[:, :,c] / image_C


	return im_mean.astype('uint8')
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


	# model_in = './data/ex_YOLO/models/model_default.h5'
	# metadata_in = './data/ex_YOLO/models/metadata_default.txt'
	# filename_image = './data/ex_detector/bike/Image.png'
	# D = detector_YOLO3.detector_YOLO3(model_in, metadata_in)
	# D.process_file(filename_image, './data/output/res_yolo.jpg')

	im = remove_bg('./images/ex_bgremoval/')
	cv2.imwrite(folder_out+'clear.jpg',im)
