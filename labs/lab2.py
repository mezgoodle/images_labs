import time

import cv2 as cv
import numpy as np
import imutils

face_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_smile.xml')
eye_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_eye.xml')
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
scaling_factor = 0.7

image = cv.imread('../data/people.jpg')
image = cv.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
gray_filter = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces_coords = face_cascade.detectMultiScale(gray_filter, minNeighbors=1)
people_recognitions = hog.detectMultiScale(image, winStride=(8, 8), padding=(30, 30))

for(x, y, width, height) in faces_coords:
    cv.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    roi_gray = gray_filter[y:y+height, x:x+width]
    roi_color = image[y:y+height, x:x+width]

    smile = smile_cascade.detectMultiScale(roi_gray)
    eye = eye_cascade.detectMultiScale(roi_gray)

    for (sx, sy, sw, sh) in smile:
        cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
    for (ex, ey, ew, eh) in eye:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

cv.imshow('Soccer team', image)
cv.waitKey(0)
cv.destroyAllWindows()
print(f'Total people: {len(people_recognitions[0])}')

camera = cv.VideoCapture(0)

while True:
    _, image = camera.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces_coords = face_cascade.detectMultiScale(gray, 1.3, 4)
    for(x, y, width, height) in faces_coords:
        cv.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    cv.imshow('Face', image)
    key = cv.waitKey(1)

    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
