# Standalone script for creating warped images, for documentation

import math
import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
import os

input = cv.imread('colours_angle_labels.png')
cv.imshow('input',input)

input_mod = copy.deepcopy(input)
input_mod[0:400] = input[1200:1600]
input_mod[400:1600] = input[0:1200]

cv.imshow('input_mod',input_mod)

warp_flags = cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR + cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
radius = 400
warped_image = cv.warpPolar(input_mod, center=((2*radius-1)/2, (2*radius-1)/2), maxRadius=radius,
                            dsize=(2 * radius, 2 * radius),
                            flags=warp_flags)

cv.imshow('warped_image',warped_image)

cv.waitKey(0)