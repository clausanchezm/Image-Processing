import numpy as mp
import cv2 as c
import scipy.fftpack

image = c.imread('images project 2/gulls.jpg', 0)
imDCT = scipy.fftpack.dct(scipy.fftpack())

c.imshow('image', image)
c.waitKey(0)
c.destroyAllWindows()
