import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise

# Filter that removes salt and pepper noise
def median_filter(image, size):
    output = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            values = []
            for i in range(int(max(0, x-(size-1)/2)), int(min(image.shape[0]-1, x + (size-1)/2))):
                for j in range(int(max(0, y-(size-1)/2)), int(min(image.shape[1]-1, y + (size-1)/2))):
                    values.append(image[i][j])
            val = np.median(values)
            output[x][y] = val
    return output

image = cv2.imread('pizza1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sp_image = random_noise(image, mode='s&p', seed=10001)
sp_image = (255*sp_image).astype(np.uint8)

plt.hist(image.ravel(), 256, [0, 256])
plt.show()

plt.hist(sp_image.ravel(), 256, [0, 256])
plt.show()

med_image = median_filter(sp_image, 5)  # change size of filter here

cv2.imshow('original', image)
cv2.imshow('salt and pepper', sp_image)
cv2.imshow('median salt and pepper', med_image)
cv2.waitKey(0)
cv2.destroyAllWindows()