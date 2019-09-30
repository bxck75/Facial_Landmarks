# Facial landmarks with dlib, OpenCV, and Python

# import the necessary packages
import datetime, imageio,argparse, imutils, time, dlib, cv2, PIL,os
from imutils import face_utils
from imutils.video import VideoStream
from testing.mark_detector import MarkDetector as mark_detector
import numpy as np

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
        print(face_utils.FACIAL_LANDMARKS_IDXS.items()) # DEBUG
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
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", clone)
            # cv2.waitKey(0)
            print((name, (i, j)))

        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)

# def _process_frame(image_path):
        image = cv2.imread(image_path[1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_captured = []        
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(128, 128)
        )
        # print(dir(faceCascade.detectMultiScale))
        print("[INFO] Found {0} Faces.".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            faces_captured.append(roi_color) 
            cv2.imwrite('faces/'+str(w) + str(h) + '_faces.jpg', roi_color)
        status = cv2.imwrite('faces_detected.jpg',  faces_captured)
        # status = cv2.imwrite('faces_detected.jpg', image)
        return status

# def process_frame(frame, detector, predictor):
    start = cv2.getTickCount()
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
        marks = mark_detector.detect_marks_keras(frame,shape)
        cv2.imshow("Image", marks)

def show_roi_detection(image, detector, predictor):
    #make a grayscale for better detecting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    faceboxes = mark_detector.extract_cnn_facebox(image)
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


    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print(rect)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # mijnface = image[y:y + h, x:x + w]


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image

        # clone = mijnface.copy()
        # cv2.imshow("Output", clone)
        # cv2.waitKey(0)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (1, 1, 1), 0)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (2, 255, 20), 3)
        # mijnface = image[(1262, 474), (1369, 582)]
        print(x,y,w,h)

        mijnface = image[(x, y), (x + w, y + h)]
        for (x, y) in shape:
            cv2.circle(mijnface, (x, y), 3, (200, 0, 255), 3)

        cv2.imshow("Output", image)
        cv2.imshow("Output", mijnface)
        cv2.waitKey(0)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)

def show_raw_detection(image, detector, predictor):
    #make a grayscale for better detecting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (1, 1, 1), 0)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 255, 255), 3)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)

def main(args):
    print(args)
    # The facial landmark predictor over an image
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    image = cv2.imread(args['image'])
    image = imutils.resize(image, width=2000)
    if args['draw']:
        draw_individual_detections(image, detector, predictor)
    else:
        # _process_frame(image)
        # show_raw_detection(image, detector, predictor)
        show_roi_detection(image, detector, predictor)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor",default='shape_predictor_68_face_landmarks.dat')
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument('--draw', nargs='?', const=True, type=bool, default=False, help="Fill landmarks")
    ap.add_argument('--cam', nargs='?', const=True, type=bool, default=False, help="Use Cam")
    
    # Init main with parsed vars
    main(vars(ap.parse_args()))
    # open till q is pressed
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()



