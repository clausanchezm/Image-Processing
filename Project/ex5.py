import cv2
import numpy as np
import math
from numpy.fft import fft2
from matplotlib import pyplot as plt


# assuming its already in grey colout s
def getMagSpec(image):
    ftIm = np.fft.fftshift(np.fft.fft2(image))
    magnitudeSpec = np.multiply(20, np.log(np.abs(ftIm)))
    return magnitudeSpec


def getPS2D(image):
    magS = getMagSpec(image)
    powerSpec = np.power(np.abs(magS), 2)
    return powerSpec


def getPS1D(image):
    w, h = image.shape
    ps2D = getPS2D(image)
    return ps2D[int(w/2), :]


def getPS3D(image):
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    return x, y


# from LAB4
def butterworthLPF(d0, n1, n2, n):
    k1, k2 = np.meshgrid(np.arange(-round(n2 / 2) + 1, math.floor(n2 / 2) + 1),
                         np.arange(-round(n1 / 2) + 1, math.floor(n1 / 2) + 1))
    d = np.sqrt(k1 ** 2 + k2 ** 2)
    h = 1 / (1 + (d / d0) ** (2 * n))
    return h


def gaussLPF(d0,n1,n2):
    k1, k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1),
                        np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    h = np.exp(-(d**2)/2/d0/d0)
    return h


def apply_filter(img,filter_mask):
    # trasnform to fourier
    f = np.fft.fftshift(np.fft.fft2(img))
    # apply filter
    f1 = f * filter_mask
    # shift back
    x1 = np.fft.ifft2(np.fft.ifftshift(f1))
    # normalize
    norm = abs(x1) / 255
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img,cmap="gray")
    ax1.set_title("Original")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(norm, cmap="gray")
    ax2.set_title("Transformed")
    plt.show()


image = cv2.imread('images project1/fog.jpg')
image = cv2.resize(image, (400, 400))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
info = np.iinfo(image.dtype)
# normalizing
doubleGrey = image.astype(np.float64)/info.max
height, width = doubleGrey.shape
x = np.linspace(5, 7, width)
# adding noise
transformX = np.cos(np.pi*x*24)*0.2
noise = doubleGrey + transformX

# power spectrum of noisy image
ps2noisy = getPS2D(noise)
ps2og = getPS2D(image)

ps1Dnoisy = getPS1D(noise)
ps1Dog = getPS1D(image)

# x3D, y3D = getPS3D(noise)
# fig = plt.figure(figsize =(14, 9))
# ax = plt.axes(projection ='3d')
# ax.plot_surface(x3D, y3D, noise.T, cmap=plt.cm.coolwarm, linewidth = 0)
# plt.title("3D power spectrum noisy image")
# plt.show()
# plt.imshow(ps2noisy, cmap="gray")
# plt.title('noisy power spectrum')
# plt.show()
# plt.imshow(ps2og, cmap='gray')
# plt.title('original power spectrum')
# plt.show()


d0 = 40
denoised = butterworthLPF(d0, noise.shape[0], noise.shape[1], 1)
# cv2.imshow('denoised B', denoised)
apply_filter(noise, denoised)
psDenoisedB = getPS2D(denoised)
plt.imshow(psDenoisedB, cmap="gray")
plt.title('denoised with ButterWorth PS')
plt.show()

# denoisedG = gaussLPF(d0, noise.shape[0], noise.shape[1])
# # cv2.imshow('denoised G', denoisedG)
# apply_filter(noise, denoisedG)
# psDenoisedG = getPS1D(denoisedG)
# plt.plot(psDenoisedB)
# # plt.plot(ps1Dog)
# # plt.plot(ps1Dnoisy)
# # plt.title('1DPS of Images 12, 25, DenoisedIm')
# plt.show()


# cv2.imshow('Grey', image)
# cv2.imshow('noisy', noise)


cv2.waitKey(0)
cv2.destroyAllWindows()
