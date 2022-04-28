import cv2
import numpy as np
from numpy.fft import fft2
from matplotlib import pyplot as plt

image = cv2.imread('images project1/fog.jpg')
greyIm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
info = np.iinfo(greyIm.dtype)
doubleGrey = greyIm.astype(np.float64)/info.max
#
#
height, width = doubleGrey.shape
x = np.linspace(0, 1, width)
y = np.linspace(0, height-1, height)
#
#
#transformX = np.cos(x*np.pi)
#transformY = np.cos(y*np.pi)
#
# #transform1 = np.reshape(transformX, (doubleGrey.shape[0], doubleGrey.shape[1]))
# #transform2 = np.reshape(transformY, (doubleGrey.shape[0], 1))
# #transform = transform1 * transform2
noise = doubleGrey + np.cos(np.pi*x*32)*0.25
#
# #temp = fft2(doubl e(image))
cv2.imshow('ogGrey', greyIm)
cv2.imshow('Grey', image)
cv2.imshow('fftPhoto ', noise)

fft2Im = np.fft.fft2(greyIm)

#spec_fft = np.fft.fftshift(fft2Im)
#back= np.real(np.fft.ifft2(np.fft.ifftshift(spec_fft)))
#cv2.imshow('spec', spec_fft)
#cv2.imshow('back',back)
#powerSpec = np.abs(fft2Im)**2


#plt.plot(powerSpec)
#plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()