#######################import the necessary packages#############################################
import datetime, imageio,argparse, imutils, time, dlib, cv2, PIL,os
from imutils.video import VideoStream
from imutils import face_utils
from testing.utils import smoothL1, relu6, DepthwiseConv2D, mask_weights
from testing.mark_detector import MarkDetector
from PIL import Image

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
 
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", default='shape_predictor_68_face_landmarks.dat',
# 	help="path to facial landmark predictor" )
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
 
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
 
# # load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=500)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# # detect faces in the grayscale image
# rects = detector(gray, 1)



def show_raw_detection(image, detector, predictor):
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	 
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	 
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
	 
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 4, (0, 0, 255), 2)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			clone = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

			# show the particular face part
			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)
			# cv2.waitKey(0)

		# visualize all facial landmarks with a transparent overlay
		output1 = face_utils.visualize_facial_landmarks(image, shape)
		output2 = face_utils.visualize_facial_landmarks(roi, shape)
		output3 = face_utils.visualize_facial_landmarks(clone, shape)
		cv2.imshow("Image", output1)
		cv2.imshow("ROI", output2)
		cv2.imshow("Clone", output3)
		cv2.waitKey(0)



	# 	#save face
	# 	cv2.imshow("Output", bounding_rect)
	# 	cv2.imwrite('faces/grabbed_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg', bounding_rect) 
	# 	# cv2.imwrite('faces/grabbed_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg',image)
	# 	print('image saved to : faces/grabbed_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg')
	# # show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image)
	# # cv2.imwrite('result/image_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg',image)
	# print('image saved to : result/image_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg')
	# cv2.waitKey(0)



def draw_individual_detections(image, detector, predictor):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
						0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		# show the particular face part
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)

		cv2.imwrite('faces/grabbed_'+str(x)+'_'+str(y)+'_face_'+str(i)+'.jpg',roi)
		# visualize all facial landmarks with a transparent overlay
		output = face_utils.visualize_facial_landmarks(image, shape)
		cv2.imshow("Image", output)



def main1():

	#######################householdings: remove old frame images####################################
	os.system('sudo rm -r test_images/*')

	#######################Input parse###############################################################
	# constructing the argument parse and parsing the arguments
	ap = argparse.ArgumentParser()
	# --shape-predictor : The path to dlib’s pre-trained facial landmark detector.'
	ap.add_argument("-p", "--shape-predictor",default = 'shape_predictor_68_face_landmarks.dat',
						help = "path to facial landmark detector")
	# --video-file input the video file to predict on
	ap.add_argument("-c", "--camera", type = bool, default = True, 
						help = "bool switch wheather videofile or camera should be used")
	# --camera : True = Use Camera ,False = --video-file input
	ap.add_argument("-v", "--video-file", type = bool, default = None, 
						help = "if --camera is set to False  we need to define the input video file.")
	# parse 
	args = vars(ap.parse_args())

	#######################Landmark predictor########################################################
	# Load the detector data
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	# exctract the shapes from data file
	print("[INFO] extracting facial landmark shapes from data...")
	predictor = dlib.shape_predictor(args["shape_predictor"])
	# initialize the video stream and allow the camera sensor to warm-up
	print("[INFO] camera sensor warming up...")
	# start the Videostream
	vs = VideoStream(0).start()
	# wait for cam to startup
	time.sleep(1.0)

	mark_detector = MarkDetector()
	# loop over the frames from the video stream
	while True:
		# start timer
		start = cv2.getTickCount()
		# grab the frame from the threaded video stream, resize it to
		frame = vs.read()
		# resize it to custom res
		frame = imutils.resize(frame, width = 800, height=600)
		#flip the frame
		frame = cv2.flip(frame, 1)
		# Detect in faceboxes
		faceboxes = mark_detector.extract_cnn_facebox(frame)
		# check if any faceboxes
		if faceboxes is not None:
			# for every facebox...
			for facebox in faceboxes:
				# Detect landmarks from image of 64X64 with grayscale.
				face_img = frame[facebox[1]: facebox[3],facebox[0]: facebox[2]]
				# draw box
				cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)

				face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
				face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
				face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)

				# timing
				land_start_time = time.time()
				marks = mark_detector.detect_marks_keras(face_img0)
				# marks *= 255
				marks *= facebox[2] - facebox[0]
				marks[:, 0] += facebox[0]
				marks[:, 1] += facebox[1]
				# Draw Predicted Landmarks
				mark_detector.draw_marks(frame, marks, color=(255, 255, 255), thick=2)
				print(marks)

		# make gray scale image for better detection
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0) 
		#loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then convert the
			shape = predictor(gray, rect)
			# facial landmark (x, y)-coordinates to a NumPy array
			shape = face_utils.shape_to_np(shape)
			# loop over the (x, y)-coordinates for the facial landmarks
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
		
			fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
			cv2.putText(frame, '%.1ffps'%(1/fps_time) , (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
			land_start_time = time.time()
			marks = mark_detector.detect_marks_keras(face_img0)

		#show the frame
		cv2.imshow("Frame", frame)
		cv2.imwrite('test_images/test_'+str(x)+'_'+str(y)+'.jpg',frame)
		key = cv2.waitKey(20) 



def main(args):

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=8000)

	if args['draw']:
		draw_individual_detections(image, detector, predictor)
	else:
		show_raw_detection(image, detector, predictor)

	key = cv2.waitKey(20) 

	#if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		cv2.destroyAllWindows()
		# vs.stop()

if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor",default='shape_predictor_68_face_landmarks.dat')
	ap.add_argument("-i", "--image", required=True, help="path to input image")
	ap.add_argument('--draw', nargs='?', const=True, type=bool, default=False, help="Fill landmarks")

	args = vars(ap.parse_args())
	main(args)
#######################householdings:closing video stream####################################
	
	