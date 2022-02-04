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

# Save image
cv.imwrite('data/UtherArtGray.jpg', image_gray)

# Image shape
(height, width, depth) = image.shape
print(f'This image has height: {height}px, width: {width}px and depth: {depth}')

# Get pixel color
(blue, green, red) = image[250, 250]
print(f'This pixel has blue: {blue}, green: {green} and red: {red} intensity')
