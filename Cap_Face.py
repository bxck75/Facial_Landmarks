import cv2
import dlib
import numpy
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import imutils

PREDICTOR_PATH = "./shape_predictor_81_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = 'assets/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

webcam = WebcamVideoStream(0).start()
fps = FPS().start()

while True:
    im = webcam.read()
    im = imutils.resize(im, width=400)

    faces = cascade.detectMultiScale(im, 1.3, 5)
    if len(faces) != 0:
        for (x, y, w, h) in faces.astype(long):
            rect = dlib.rectangle(x, y, x + w, y + h)
            #cv2.imwrite('face.png', rect)
            get_landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

        for idx, point in enumerate(get_landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
           # cv2.drawContours(im, [pos], -1, (0, 255, 0), 2)
#            hullIndex = cv2.convexHull(pos, returnPoints=False)
#            cv2.imwrite('face.png', hullIndex)
            cv2.circle(im, pos, 3, color=(0, 255, 255))
    cv2.imshow('Result', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

cv2.destroyAllWindows()