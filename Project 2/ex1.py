import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


def toPlot(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def addMotionBlur(image, alpha, beta):
    x, y, z = image.shape
    ft = np.fft.fft2(image)
    [u, v] = np.mgrid[-x/2: x/2, -y/2: y/2]
    u = 2*u/x
    v = 2*v/y
    h = np.multiply(u, alpha) + np.multiply(v, beta)
    a = -1j*(h * np.pi)
    H = np.multiply(np.sinc(h), np.exp(a))
    # thisis in fourier transform
    G = ft
    G[:,:, 0] = ft[:,:,0] * H
    G[:,:, 1] = ft[:,:,1] * H
    G[:,:, 2] = ft[:,:,2] * H
    g = np.fft.ifft2(G)
    # g = abs(g)/255
    return g


def addRandomNoise(image, mode, mean, var):
    xn = random_noise(np.abs(image).astype(np.uint8), mode = mode, mean = mean, var = var)
    XN = np.fft.fft2(xn)
    return xn


bird = c.imread('images project 2/bird.jpg')
# pyplot reads RGB and cv2 in BGR
bird = c.cvtColor(bird, c.COLOR_BGR2RGB)
toPlot(bird, "original bird")
bird = bird.astype(np.double)
blur = addMotionBlur(bird, 0.6, 0.15)
toPlot(np.abs(blur)/255, "blurred, A= 0.6, B=0.15")
noisy = addRandomNoise(blur, 'gaussian', 0, 0.002)
toPlot(noisy.astype(np.double), "Gaussian Noise (m=0, v =0.002)")

