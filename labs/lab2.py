import cv2 as cv
import numpy as np
import imutils

face_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_smile.xml')
eye_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_eye.xml')
scaling_factor = .5

image = cv.imread('../data/zuckerberg.jpg')
image = cv.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
gray_filter = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces_coords = face_cascade.detectMultiScale(gray_filter, minNeighbors=1)

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

cv.imshow('Zuckerberg\'s family', image)
cv.waitKey(0)
print(f'Total faces: {len(faces_coords)}')
