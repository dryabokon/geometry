import cv2
import sys
import numpy
# ----------------------------------------------------------------------------------------------------------------------
sys.path.append('../')
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
filename_in = 'images/ex13/01.jpg'
filename_out = 'images/output/res.png'
USE_CAMERA = True

# ---------------------------------------------------------------------------------------------------------------------
def demo_bg_removal():
	if USE_CAMERA:
		cap = cv2.VideoCapture(0)
	else:
		cap = []
		frame = cv2.imread(filename_in)

	fgbg = cv2.createBackgroundSubtractorMOG2()



	while (True):
		if USE_CAMERA:
			ret, frame = cap.read()

		fgmask = fgbg.apply(frame)
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
if __name__ == '__main__':
	demo_bg_removal()
