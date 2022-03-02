import cv2 as cv
import numpy as np


def process_image(image):
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    kernel_size = (5, 5)
    blur_gray = cv.GaussianBlur(grey_image, kernel_size, 0)

    low_treshold = 50
    high_treshold = 150
    edges = cv.Canny(blur_gray, low_treshold, high_treshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    image_shape = image.shape
    vertices = np.array([[(0, image_shape[0]), (960, 690), (1080, 690), (image_shape[1], image_shape[0])]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(edges, mask)

    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 30

    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)


    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return image


video_capture = cv.VideoCapture('../data/road_video.mp4')

while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        output = process_image(frame)
        cv.imshow('frame', output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv.destroyAllWindows()
