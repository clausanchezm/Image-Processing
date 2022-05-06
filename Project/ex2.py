import cv2
import numpy as np
from matplotlib import pyplot as plt

fogIm = cv2.imread("images project1/fog.jpg")
shadowsIm = cv2.imread("images project1/shadows.jpg")
shadowsIm = cv2.resize(shadowsIm, (960, 400))
fogIm = cv2.resize(fogIm, (960, 400))


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
    neg = np.abs(255 - image)
    return neg


# 2.4
# power law that alternates 0 to 1 can be the same as gamma correction with value of the pixel
def gammaCorrection(image, gamma):
    power = np.power(image, gamma)
    return power


# 2.3
# negFog = negativePointInv(fogIm)
# negShadow = negativePointInv(shadowsIm)
# cv2.imshow('negative fog ', negFog)
# cv2.imshow('negative shadows', negShadow)
# plt.hist(negFog.ravel(), 256, [0, 256])
# plt.title('Fog Negative Point Inversion')
# plt.xlabel('grayscale')
# plt.show()

# negShadow = negativePointInv(shadowsIm)
# cv2.imshow('negative shadow', negShadow)
# plt.hist(negShadow.ravel(), 256, [0, 256])
# plt.title('Shadow Negative Point Inversion')
# plt.xlabel('grayscale')
# plt.show()


cv2.imshow('original shadow', shadowsIm)
cv2.imshow('original fog', fogIm)
shadowsG2 = gammaCorrection(shadowsIm, 2)
shadowsG06 = gammaCorrection(shadowsIm, 0.6)
fogG2 = gammaCorrection(fogIm, 2)
fogG06 = gammaCorrection(fogIm, 0.9)
cv2.imshow('shadows  g =0.6', shadowsG06)
cv2.imshow('shadows  g =2', shadowsG2)
cv2.imshow('fog g=0.6', fogG06)
cv2.imshow('fog g = 2', fogG2)

plt.hist(shadowsG2.ravel(), 256, [0, 256])
plt.title('shadows gamma=2')
plt.xlabel('grayscale')
plt.show()
plt.hist(shadowsG06.ravel(), 256, [0, 256])
plt.title('shadows gamma=0.6')
plt.xlabel('grayscale')
plt.show()
plt.hist(fogG06.ravel(), 256, [0, 256])
plt.title('fog gamma=0.9')
plt.xlabel('grayscale')
plt.show()
plt.hist(fogG2.ravel(), 256, [0, 256])
plt.title('fog gamma=2')
plt.xlabel('grayscale')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
