import cv2
from tqdm import tqdm
import numpy
import pybgs as bgs
# ----------------------------------------------------------------------------------------------------------------------
#import tools_image
#import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
USE_CAMERA = False
# ---------------------------------------------------------------------------------------------------------------------
def demo_bg_removal_cam():
	if USE_CAMERA:cap = cv2.VideoCapture(0)
	else:cap = cv2.VideoCapture(source)
	#8 10 12 13
	algos = [
		bgs.FrameDifference(), bgs.StaticFrameDifference(), bgs.WeightedMovingMean(),
		bgs.WeightedMovingVariance(), bgs.AdaptiveBackgroundLearning(),
		bgs.AdaptiveSelectiveBackgroundLearning(), bgs.MixtureOfGaussianV2(),
		bgs.PixelBasedAdaptiveSegmenter(), bgs.SigmaDelta(), bgs.SuBSENSE(), bgs.LOBSTER(),
		bgs.PAWCS(), bgs.TwoPoints(), bgs.ViBe(), bgs.CodeBook(),
		bgs.FuzzySugenoIntegral(), bgs.FuzzyChoquetIntegral(), bgs.LBSimpleGaussian(),
		bgs.LBFuzzyGaussian(), bgs.LBMixtureOfGaussians(), bgs.LBAdaptiveSOM(),
		bgs.LBFuzzyAdaptiveSOM(), bgs.VuMeter(), bgs.KDE(), bgs.IndependentMultimodal()
	]

	#fgbg = bgs.ViBe()
	fgbg = algos[13]

	print(fgbg.__class__.__name__)

	while (True):
		ret, frame = cap.read()
		frame = cv2.resize(frame, (1280, 720))

		fgmask = fgbg.apply(frame)
		if len(fgmask.shape)==2:
			fgmask = cv2.cvtColor(fgmask.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)

		fgmask[:, :, 0] = 0
		fgmask[:, :, 1] = 0
		frame = cv2.cvtColor(frame.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
		frame = cv2.cvtColor(frame.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)

		result = cv2.addWeighted(frame, 0.7, fgmask, 0.3, 0)

		cv2.imshow('frame', result)
		if cv2.waitKey(1) & 0xFF == 27:
			break

	if USE_CAMERA:
		cap.release()

	cv2.destroyAllWindows()

	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_bg_removal_video(source,folder_out):
	if ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower()):mode = 'video'
	elif ('https' in source) or (source == '0'):mode = 'stream'
	else:mode = 'folder'
	fgbg = cv2.createBackgroundSubtractorMOG2()

	if mode == 'video':
		vidcap = cv2.VideoCapture(source)
		total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		for i in tqdm(range(total_frames), total=total_frames):
			success, image = vidcap.read()
			image = cv2.resize(image, (1280, 720))
			if not success: continue
			fgmask = fgbg.apply(image)
			cv2.imwrite(folder_out + 'frame_%06d.jpg'%i, fgmask)
		vidcap.release()
	elif mode == 'folder':
		filenames = tools_IO.get_filenames(source, '*.jpg,*.png')
		for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
			image = cv2.imread(folder_in+filename)
			fgmask = fgbg.apply(image)
			cv2.imwrite(folder_out+filename,fgmask)

	return
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
#source = './images/ex_optical_flow/TownCentreXVID.mp4'
#source = './images/ex_optical_flow/Naft11.mp4'
#source = './images/ex_optical_flow/Day-Praktik-1_B2.mp4'
source = './images/ex_optical_flow/Night-Mayak-5.mp4'
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	#demo_bg_removal_video(source,folder_out)
	demo_bg_removal_cam()


