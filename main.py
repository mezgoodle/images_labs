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

# Get image part as slice
# input image starting at x=200 ,y=10 and ending at x=420 ,y=200
head = image[10:200, 200:420]
cv.imshow('Head', head)
cv.waitKey(0)
cv.destroyAllWindows()

# Resize the image
resized_image = cv.resize(image, (200, 200))
cv.imshow('Resized image', resized_image)
cv.waitKey(0)

# Resize the image with the ratio
new_height = 200
ratio = width / height
new_width = int(new_height * ratio)
resized = cv.resize(image, (new_width, new_height))
print(f'Shape of the resized image with ratio: {resized.shape}')
cv.imshow('Resized image with ratio', resized)
cv.waitKey(0)
