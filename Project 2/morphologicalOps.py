import cv2 as c
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
from skimage import io


def open(image, kernel):
    eroded = c.erode(image, kernel)
    dilated = c.dilate(eroded, kernel)
    return dilated


def close(image, kernel):
    dilated = c.dilate(image, kernel)
    eroded = c.erode(dilated, kernel)
    return eroded


# openings decreases the intensity of grayscale
def granulometry(image, sizes):
    intensity = np.zeros(len(sizes))
    for i in range(len(sizes)):
        kernel = c.getStructuringElement(c.MORPH_ELLIPSE, (i+1, i+1))
        im = open(image, kernel)
        # the sum of all intensity pixels is the surface area, as opening is done with bigger ellipse, the intensities computed will be lower
        intensity[i] = np.sum(im)
    return intensity


#     plot the difference ebtween the neighbouring elements of the array
def plotDiff(intens, sizes, original):
    # intens = intens/ max(intens)
    og = np.sum(original)
    res = np.zeros((len(sizes)))
    diff = np.gradient(intens)
    for i in range(len(sizes)-1):
        res[i] = intens[i+1] - intens[i]
    plt.plot( sizes,diff, '-bo')
    plt.title('difference in surface area / radius of SE')
    plt.show()


# image passed has already been closed and opened
def countSE(imageoC, upperBound, lowerBound):
    # label identifies the white regions of the threshold im; so identifies the oranges. for oranges it is enough to count them
    label_im = label(imageoC)
    # regions: organizes the labels by ordering them all into a matrix with coors
    regions = regionprops(label_im)
    # print(len(regions)-1)
    io.imshow(label_im)

    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        # ratios UB &LB to adjust to shape of the objects counting
        if (num != 0 and (area > 10) and (convex_area / area < upperBound)
                and (convex_area / area > lowerBound)):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    count = len(masks)
    return count


# 3.1
oranges = c.imread('images project 2/oranges.jpg')
orangeTree = c.imread('images project 2/orangetree.jpg')
# 3.2
granu = c.imread('images project 2/granulometry1.jpg')

oranges = np.uint8(oranges)
ot = np.uint8(orangeTree)
granu = np.uint8(granu)

oranges = c.cvtColor(oranges, c.COLOR_BGR2RGB)
ot = c.cvtColor(ot, c.COLOR_BGR2RGB)
g = c.cvtColor(granu, c.COLOR_BGR2RGB)
orangesB = c.cvtColor(oranges, c.COLOR_RGB2GRAY)
otB = c.cvtColor(ot, c.COLOR_RGB2GRAY)
gB = c.cvtColor(g, c.COLOR_RGB2GRAY)

# twixing the param for most convinience
num, orangeT = c.threshold(orangesB, 127, 255, c.THRESH_BINARY)
num2, otT = c.threshold(otB, 135, 255, c.THRESH_BINARY)
num3, gT = c.threshold(gB, 130, 255, c.THRESH_BINARY)
c.imshow('threshold', gT)


# 3.1 COUNTING ORANGES
# using this size was suitable for both images
circleB = c.getStructuringElement(c.MORPH_ELLIPSE, (7, 7))
circleA = c.getStructuringElement(c.MORPH_ELLIPSE, (9, 9))
openO = open(orangeT, circleA)
closedO = close(openO, circleA)
openT = open(otT, circleB)
closedT = close(openT, circleB)

# countSE(closedO, 3, 0.1)

# print(countSE(closedT, 1.061, 0.90))


# 3.2 GRANULOMETRY
sizes = np.linspace(1, 60, 60)
inten = granulometry(gT, sizes)
plt.plot(sizes, inten)
plt.title('sum of intensities ')
plt.show()
plotDiff(inten, sizes, gT)


c.waitKey(0)
c.destroyAllWindows()


#  TRYING WITH MASKS AND BITWISE OPERATIONS
# def circle_structure(n):
#     struct = np.zeros((2 * n + 1, 2 * n + 1))
#     x, y = np.indices((2 * n + 1, 2 * n + 1))
#     #creating a mask ( eveything is black except the circle we just created )
#     mask = (x - n)**2 + (y - n)**2 <= n**2
#     struct[mask] = 1
#     return struct
#
#
# def granulo(image, sizes):
#     for n in sizes:
#         bit_and = c.bitwise_and(image, circle_structure(n))
#         print(bit_and)
#     return


# # another way to count circles
# detected_circles = c.HoughCircles(ot,
#                    c.HOUGH_GRADIENT, 1, 20, param1 = 70,
#                param2 = 20, minRadius = 1, maxRadius = 40)
# counter =0
# for cir in detected_circles[0, :]:
#     counter = counter + 1
#
# print('counter', counter)




