import cv2
import numpy as np

THRESHOLD_DEPTH = 100

img = cv2.imread("../processed/frame1.jpg")
print(img)
depth_img = np.zeros(shape=img.shape[1:])

img = img if depth_img > THRESHOLD_DEPTH else 0

edge_detected_image = cv2.Canny(img, 75, 200)

_, contours, hierarchy = cv2.findContours(
    edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_list = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)

    if ((len(approx) > 8) & (len(approx) < 23) & (area > 30)):
        contour_list.append(contour)

cv2.drawContours(img, contour_list, -1, (255, 0, 0), 2)
cv2.imshow(img)
