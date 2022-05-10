import cv2
import numpy as np

image = cv2.imread('pizza1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# h1 = Vertical h2 = Horizontal h3 = Diagonal
h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
h2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
h3 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

v_image = cv2.filter2D(image, -1, h1)
h_image = cv2.filter2D(image, -1, h2)
d_image = cv2.filter2D(image, -1, h3)

cv2.imshow('original', image)
cv2.imshow('vertical', v_image)
cv2.imshow('horizontal', h_image)
cv2.imshow('diagonal', d_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
