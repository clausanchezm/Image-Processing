import cv2
import matplotlib.pyplot as plt
import numpy as np


def getMagSpectrum(im):
    imG = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ftIm = np.fft.fft2(imG)
    ftIm = np.fft.fftshift(ftIm)
    #magnitude spectrum plot
    magnitudeSpec = np.multiply(20, np.log(np.abs(ftIm)))
    return magnitudeSpec


image = cv2.imread("images project1/crayons.jpg")
image = cv2.resize(image, (400, 400))
cv2.imshow('original', image)
padded = np.pad(image, ((0,0), (500,0), (0,0)), mode='constant')
cv2.imshow('padded', padded)

ogMS = getMagSpectrum(image)
paddedMS = getMagSpectrum(padded)
plt.imshow(ogMS)
plt.title('Magnitude Spectrum of original image')
plt.show()

plt.imshow(paddedMS)
plt.title('Magnitude Spectrum of Padded image ')
plt.show()
# cv2.imwrite('images project1/paddedCrayons.jpg', padded)
cv2.imshow('padded', padded)

cv2.waitKey(0)
cv2.destroyAllWindows()
