import cv2 as cv
import numpy as np


def image_to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def blur_image(image):
    kernel_size = (5, 5)
    return cv.GaussianBlur(image, kernel_size, 0)


def find_edges(image):
    low_treshold = 50
    high_treshold = 150
    return cv.Canny(image, low_treshold, high_treshold)


def find_masks(edges, image):
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    image_shape = image.shape
    vertices = np.array([[(0, image_shape[0]), (960, 690), (1080, 690), (image_shape[1], image_shape[0])]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    return cv.bitwise_and(edges, mask)


def detect_lines(masks):
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 30

    return cv.HoughLinesP(masks, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)


def process_image(image):
    grey_image = image_to_gray(image)
    blur_gray = blur_image(grey_image)
    edges = find_edges(blur_gray)
    masked_edges = find_masks(edges, image)
    lines = detect_lines(masked_edges)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return image


video_capture = cv.VideoCapture('../data/road_video.mp4')
frame_counts = 0

while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        frame_counts += 1

        output = process_image(frame)

        font = cv.FONT_HERSHEY_COMPLEX
        cv.putText(output, 'Maxim', (1300, 700), font, 2, (0, 0, 0), 2, cv.LINE_8)

        cv.imshow('frame', output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv.destroyAllWindows()
print(f'Total frame number is: {frame_counts}')
