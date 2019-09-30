#import the libraries 
import PIL.Image
import PIL.ImageDraw
import numpy as np
import face_recognition
import datetime, imageio,argparse, imutils, time, dlib, cv2, PIL,os
from imutils.video import VideoStream
from imutils import face_utils
from testing.utils import smoothL1, relu6, DepthwiseConv2D, mask_weights
from testing.mark_detector import MarkDetector

print(dir(imutils.skeletonize))
def landmark_image(image_file_path):
	# on image
	# Load the jpg file into a NumPy array
	image = face_recognition.load_image_file(image_file_path)

	# Find all the faces in the image
	face_locations_list = face_recognition.face_locations(image)
	face_landmarks_list = face_recognition.face_landmarks(image)
	face_encodings_list = face_recognition.api.face_encodings(image, known_face_locations=face_locations_list)

	# for face_location in face_locations_list:
	# 	face_encoded = face_recognition.api.face_encodings(image, known_face_locations=face_location)
	# 	face_recognition.api.compare_faces(face_encodings_list, face_encoded)
	# 	print(face_compaired)

	number_of_faces = len(face_locations_list)
	print("I found {} face(s) in this photograph.".format(number_of_faces))
	number_of_landmarks = len(face_landmarks_list)
	print("I found {} landmarkset(s) in this photograph.".format(number_of_landmarks))
	number_of_encodings = len(face_encodings_list)
	print("I encoded {} face(s) in this photograph.".format(number_of_encodings))

	# Load the image into a Python Image Library object so that we can draw on top of it and display it
	pil_image = PIL.Image.fromarray(image)

	for face_location in face_locations_list:

		# Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
		top, right, bottom, left = face_location
		print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

		# Let's draw a box around the face
		draw = PIL.ImageDraw.Draw(pil_image)
		draw.rectangle([left, top, right, bottom], outline="red")
	
	print(face_landmarks_list)
	for face_landmarks in face_landmarks_list:
		# print(face_landmarks)
		# Loop over each facial feature (eye, nose, mouth, lips, etc)
		for name, list_of_points in face_landmarks.items():
			# print(list_of_points)
			hull = np.array(face_landmarks[name])
			hull_landmark = cv2.convexHull(hull)
			cv2.drawContours(image, hull_landmark, -1, (0, 255, 0), 3)
			# draw.rectangle([left, top, right, bottom], outline="red")
			print(name)
			print(face_landmarks[name][0])
			cv2.circle(image, face_landmarks[name][0], 10, (0, 255, 0), 3)
	# Display the image on screen
	pil_image.show()


def landmark_video():

	#videostream
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()

		frame = cv2.resize(frame, (0,0), fx=1, fy=1)

		# Find all facial features in all the faces in the video
		face_landmarks_list = face_recognition.face_landmarks(frame)

		for face_landmarks in face_landmarks_list:
			# Loop over each facial feature (eye, nose, mouth, lips, etc)
			for name, list_of_points in face_landmarks.items():

				hull = np.array(face_landmarks[name])
				hull_landmark = cv2.convexHull(hull)
				cv2.drawContours(frame, hull_landmark, -1, (0, 255, 0), 3)


		cv2.imshow("Frame", frame)

		ch = cv2.waitKey(1)
		if ch & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':

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
	###################################################

	landmark_image('samples/oega.jpg')