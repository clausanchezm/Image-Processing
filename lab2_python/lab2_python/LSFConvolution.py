import cv2
import numpy as np


# method for convolution
def convolve(image, kernel):
    output = np.zeros_like(image)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  #zero padding
    for x in range(1, output.shape[0]):
        for y in range(1, output.shape[1]):
            roi = image[x-1:x+1+1, y-1:y+1+1]
            val = (roi * kernel).sum()
            output[x][y] = val
    return output

image = cv2.imread('pizza1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
h2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
h3 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

v_image = convolve(image, h1)
h_image = convolve(image, h2)
d_image = convolve(image, h3)

cv2.imshow('original', image)
cv2.imshow('vertical', v_image)
cv2.imshow('horizontal', h_image)
cv2.imshow('diagonal', d_image)
cv2.waitKey(0)
cv2.destroyAllWindows()