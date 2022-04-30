import cv2
import numpy as np
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


def filterNoisePS(fftIm, ps2D):
    indices = ps2D > 100
    PSclean = ps2D * indices
    fftIm = indices * fftIm
    inv = np.real(np.fft.ifft(np.fft.ifftshift(fftIm)))
    # back= np.real(np.fft.ifft2(np.fft.ifftshift(spec_fft)))
    return PSclean, inv


image = cv2.imread('images project1/fog.jpg')
image = cv2.resize(image, (400, 400))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
info = np.iinfo(image.dtype)
# normalizing
doubleGrey = image.astype(np.float64)/info.max
height, width = doubleGrey.shape
x = np.linspace(0, 1, width)
# adding noise
transformX = np.cos(np.pi*x*25)*0.2
noise = doubleGrey + transformX

# power spectrum of noisy image
ps2noisy = getPS2D(noise)
ps2og = getPS2D(image)

ps1Dnoisy = getPS1D(noise)

x3D, y3D = getPS3D(noise)
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface(x3D, y3D, noise.T, cmap=plt.cm.coolwarm, linewidth = 0)
plt.title("3D power spectrum")
plt.show()
plt.imshow(ps2noisy, cmap="gray")
plt.title('noisy power spectrum')
plt.show()
plt.imshow(ps2og, cmap='gray')
plt.title('original power spectrum')
plt.show()
plt.plot(ps1Dnoisy)
plt.title('1D NOISY power spectrum')
plt.show()


psClean, inverse = filterNoisePS(image, getPS2D(image))
plt.imshow(psClean, cmap='gray')
plt.title('denoised power spectrum')
plt.show()
# plt.imshow(inverse)
#spec_fft = np.fft.fftshift(fft2Im)
#back= np.real(np.fft.ifft2(np.fft.ifftshift(spec_fft)))
#cv2.imshow('spec', spec_fft)
#cv2.imshow('back',back)
#powerSpec = np.abs(fft2Im)**2


#plt.plot(powerSpec)
#plt.show()

cv2.imshow('Grey', image)
cv2.imshow('fftPhoto ', noise)
cv2.imshow('denoised', inverse)

cv2.waitKey(0)
cv2.destroyAllWindows()
