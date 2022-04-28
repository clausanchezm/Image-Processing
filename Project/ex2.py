import cv2
import numpy as np
from matplotlib import pyplot as plt

fogIm = cv2.imread("images project1/fog.jpg")
shadowsIm = cv2.imread("images project1/shadows.jpg")


# # 2.1
# plt.hist(fogIm.ravel(), 256, [0, 256])
# plt.title('Fog')
# plt.xlabel('gray scale')
# plt.show()
#
# plt.hist(shadowsIm.ravel(), 256, [0, 256])
# plt.title('Shadows')
# plt.xlabel('gray scale')
# plt.show()


# 2.2
def negativePointInv(image):
    height, width, _ = image.shape
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            pixel = image[i, j]
            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]
            pixel[2] = 255 - pixel[2]
            pixel[i, j] = pixel
    return pixel


# 2.4
# power law that alternates 0 to 1 can be the same as gamma correction with value of the pixel
def gammaCorrection(image, gamma):
    height, width, _ = image.shape
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            pixelVal = image[i, j]
            pixelVal[0] = np.power(pixelVal[0], gamma)
            pixelVal[1] = np.power(pixelVal[1], gamma)
            pixelVal[2] = np.power(pixelVal[2], gamma)
            image[i, j] = pixelVal
    return image


# 2.3
# negFog = negativePointInv(fogIm)
# cv2.imshow('negative fog ', negFog)
# plt.hist(negFog.ravel(), 256, [0, 256])
# plt.title('Fog Negative Point Inversion')
# plt.xlabel('pixels')
# plt.show()

# negShadow = negativePointInv(shadowsIm)
# cv2.imshow('negative shadow', negShadow)
# plt.hist(negShadow.ravel(), 256, [0, 256])
# plt.title('Shadow Negative Point Inversion')
# plt.xlabel('pixels')
# plt.show()

cv2.imshow('original shadow', shadowsIm)
powerIm = gammaCorrection(shadowsIm, 2)

powerImLESS = gammaCorrection(shadowsIm, 0.6)
cv2.imshow('powerIMageLESS', powerImLESS)
cv2.imshow('powerIMage', powerIm)
# cv2.imshow('original fog', fogIm)

cv2.waitKey(0)
cv2.destroyAllWindows()
