import cv2 as cv
import numpy as np
import imutils

face_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
scaling_factor = .5

frame = cv.imread('../data/zuckerberg.jpg')
frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
faces_coords = face_cascade.detectMultiScale(frame, minNeighbors=5)

for(x, y, width, height) in faces_coords:
    cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

cv.imshow('Zuckerberg\'s family', frame)
cv.waitKey(0)
print(f'Total faces: {len(faces_coords)}')
