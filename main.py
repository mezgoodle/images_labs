import numpy as np
import cv2 as cv
import imutils

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
cv.destroyAllWindows()

# Rotate the image
center = (width // 2, height // 2)
matrix = cv.getRotationMatrix2D(center, angle=-90, scale=1.0)
rotated_image = cv.warpAffine(image, matrix, (width, height))
cv.imshow('Rotated image', rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred_image = cv.GaussianBlur(image, (27, 27), 0)
cv.imshow('Blurred image', blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Stack images
resized = imutils.resize(image, width=288)
resized_blurred = imutils.resize(blurred_image, width=288)
stack_images = np.hstack((resized, resized_blurred))
cv.imshow('Stack image', stack_images)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a rectangle
copy = image.copy()
cv.rectangle(copy, (200, 10), (400, 230), (0, 0, 255), 2)
cv.imshow('Rectangle', copy)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a line
copy = image.copy()
cv.line(copy, (200, 10), (400, 230), (0, 0, 255), 2)
cv.imshow('Line', copy)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a polylines
copy = image.copy()
points = np.array([[200, 130], [300, 230], [400, 130], [300, 10]])
cv.polylines(copy, np.int32([points]), 1, (0, 0, 255), 2)
cv.imshow('Polylines', copy)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a circle
copy = image.copy()
cv.circle(copy, (300, 130), 100, (0, 0, 255), 2)
cv.imshow('Circle', copy)
cv.waitKey(0)
cv.destroyAllWindows()

# Put a text
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(image, 'Uther', (10, 300), font, 4, (255, 255, 255), 2, cv.LINE_8)
cv.putText(image, 'Lightbringer', (10, 400), font, 2, (255, 255, 255), 2, cv.LINE_8)
cv.imshow('Text', image)
cv.waitKey(0)
cv.destroyAllWindows()
