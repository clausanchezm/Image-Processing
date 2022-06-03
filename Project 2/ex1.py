import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


def toPlot(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def fourierTrans(image):
    # ft1, ft2, ft3 = c.split(image)
    ft1 = np.fft.fftshift(np.fft.fft2(image[:, :, 0]))
    ft2 = np.fft.fftshift(np.fft.fft2(image[:, :, 1]))
    ft3 = np.fft.fftshift(np.fft.fft2(image[:, :, 2]))
    return ft1, ft2, ft3

def inv_fourierT(img):
    ft1 = np.abs(np.fft.ifft2(img[:,:,0]))
    ft2 = np.abs(np.fft.ifft2(img[:,:,1]))
    ft3 = np.abs(np.fft.ifft2(img[:,:,2]))
    return c.merge([ft1,ft2,ft3])

def inv_fourierT(c1, c2, c3 ):
    ft1 = np.abs(np.fft.ifft2(c1))
    ft2 = np.abs(np.fft.ifft2(c2))
    ft3 = np.abs(np.fft.ifft2(c3))
    return ft1, ft2, ft3


def addMotionBlur(image, alpha, beta):
    x, y, z = image.shape
    ft1, ft2, ft3 = fourierTrans(image)
    [u, v] = np.mgrid[-round(x / 2):round(x / 2), -round(y / 2):round(y / 2)]
    u = 2*u/x
    v = 2*v/y
    H = np.sinc((u * alpha + v * beta)) * np.exp(-1j * np.pi * (u * alpha + v * beta))
    # this is in fourier transform
    G1 = ft1 * H
    G2 = ft2 * H
    G3 = ft3 * H
    ft1, ft2, ft3 = inv_fourierT(G1,G2,G3)
    # g = inv_fourierT(G)
    g= c.merge([ft1,ft2,ft3])
    return g


def getblurringF(of1, of2, of3, bf1, bf2, bf3):
    h1 =bf1 / of1
    h2 = bf2 /of2
    h3 = bf3 /of3
    return h1, h2, h3


# in oredr to denoise the image, an approximation of the blury function is obtained by divining G blured/ OG im  both in FT
def deBlur(blurIm, im):
    ft1B, ft2B, ft3B = fourierTrans(blurIm)
    ft1O, ft2O, ft3O = fourierTrans(im)
    h1, h2, h3 = getblurringF(ft1O, ft2O, ft3O, ft1B, ft2B, ft3B)
    dBft1 = ft1B/ h1
    dBft2 = ft2B / h2
    dBft3 = ft3B / h3
    dBi1, dBi2, dBi3 = inv_fourierT(dBft1,dBft2, dBft3)
    deBlu = c.merge([dBi1, dBi2, dBi3])
    return deBlu


# mode being gaussian
def addRandomNoise(image, mode, mean, var):
    xn = random_noise(np.abs(image).astype(np.uint8), mode = mode, mean = mean, var = var)
    return xn

def mmseFilter(noisy , og, k_bool):
    nFt1, nFt2, nFt3 = fourierTrans(noisy)
#     approximating the noise
    noise = og - noisy
    snn1, snn2, snn3 = fourierTrans(noise)
    sf1, sf2, sf3 = fourierTrans(og)
    # h1, h2, h3 = getblurringF(sf1, sf2, sf3, nFt1, nFt2, nFt3)
    h1= 1
    h2 =1
    h3= 1
    # computing the noise power spectrum at each channel
    snn1P = np.abs(snn1) ** 2
    snn2P = np.abs(snn2) ** 2
    snn3P = np.abs(snn3) ** 2
    sf1P = np.abs(sf1) ** 2
    sf2P = np.abs(sf2) ** 2
    sf3P = np.abs(sf3) ** 2
    # either using K approx or not
    if k_bool:
        k = np.mean([np.mean(snn1 / sf1), np.mean(snn2 / sf2), np.mean(snn3 / sf3)])
        k1 = np.abs(h1) ** 2 + k
        k2 = np.abs(h2) ** 2 + k
        k3 = np.abs(h3) ** 2 + k
        Hw1 = (np.conj(h1) * h1) / k1
        Hw2 = (np.conj(h2) * h2) / k2
        Hw3 = (np.conj(h3) * h3) / k3
    else:
        den1 = np.abs(h1) ** 2 + (snn1P / sf1P)
        den2 = np.abs(h2) ** 2 + (snn2P / sf2P)
        den3 = np.abs(h3) ** 2 + (snn3P / sf3P)
        Hw1 = (np.conj(h1) * h1) / den1
        Hw2 = (np.conj(h2) * h2)/ den2
        Hw3 = (np.conj(h3) * h3)/ den3

    f1 = Hw1 * snn1
    f2 = Hw2 * snn2
    f3 = Hw3 * snn3

    i1, i2, i3 = inv_fourierT(f1, f2, f3)
    O = c.merge([i1, i2, i3])
    return O



bird = c.imread('images project 2/bird.jpg')
# pyplot reads RGB and cv2 in BGR
# bird = c.cvtColor(bird, c.COLOR_BGR2RGB)
bird = bird/ np.max(bird)
# bird = bird.astype(np.double)
#
blur = addMotionBlur(bird, 9, 7)
dBlurred = deBlur(blur, bird)
noisyBIm = addRandomNoise(blur, 'gaussian', 0, 0.03)
noiseIm = addRandomNoise(bird, 'gaussian', 0, 0.03)
# apply the same procedure, H(u,v) now also means the additive gaussian noise
dBnoisy = deBlur(noisyBIm, bird)
mmnseK = mmseFilter(noiseIm, bird, True)
mmnse = mmseFilter(noiseIm, bird, False)
mmnse = mmnse / np.max(mmnse)
mmnseK = mmnseK / np.max(mmnseK)
c.imshow("blur.jpg",blur)
c.imshow("blurNoise.jpg",noisyBIm)
c.imshow("noise.jpg",noiseIm)
c.imshow("deBlurredM.jpg",dBlurred)
c.imshow('deBlurredMN.jpg', dBnoisy)
c.imshow('deBluredMNSEK.jpg', mmnseK)
c.imshow('deBluredMNSE.jpg', mmnse)
c.imwrite("blur.jpg",blur/ np.max(blur) * 255)
c.imwrite("blurNoise.jpg", noisyBIm / np.max(noisyBIm) * 255)
c.imwrite("noise.jpg",noiseIm / np.max(noiseIm) * 255)
c.imwrite("deBlurredM.jpg",dBlurred/ np.max(dBlurred) * 255)
c.imwrite('deBlurredMN.jpg', dBnoisy/ np.max(dBnoisy) * 255)
c.imwrite('deBluredMNSEK.jpg', mmnseK* 255)
c.imwrite('deBluredMNSE.jpg', mmnse* 255)

# c.waitKey(0)
# c.destroyAllWindows()

