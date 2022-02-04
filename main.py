import numpy as np
import cv2 as cv

image_path = 'data/UtherArt.jpg'

image = cv.imread(image_path)
image_gray = cv.imread(image_path, 0)

# Show image
cv.imshow('Uther', image)
cv.imshow('Uther Gray', image_gray)
cv.waitKey(0)
cv.destroyAllWindows()
