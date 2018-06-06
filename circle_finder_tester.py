%matplotlib inline
import matplotlib.pyplot as plt
import os
import pims
import numpy as np
import pandas as pd
import h5py
import sys
import pathlib
import multiprocessing as mp
import cv2 as cv
import skimage as sk

p = pathlib.Path('/Users/ericdeal/Dropbox (MIT)/3_postdoc/projects/sed_transport/1_data/0_main_feed_exp_data/_2017_exps/_data/glass_beads/')

def hough(img):
    # r = [20, 30]
    # r = [10, 20]
    # rings_test = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 3, 5, param1=50, param2=40, minRadius=r[0], maxRadius=r[1])
    # rc = int(np.median(rings_test[:,:,2]))
    rc = 13
    dr = 2
    rings = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 3, int(rc*.8), param1=40, param2=50, minRadius=int(rc-dr), maxRadius=int(rc+dr)).squeeze()
    print(rc, np.mean(rings[:,2]))

    img = sk.color.gray2rgb(img)
    if rings is not None:
        rings = rings.astype(int)
        for y, x, r in zip(rings[:,0], rings[:,1], rings[:,2]):
            # draw the center of the circle
            cv.circle(img, (y,x), r, (200,20,20), 1, cv.LINE_AA)

    fig = plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

i = 12
j = 0
video = pims.Video(str(list(p.rglob('exp_transport_stage_%i/manta/*.5.mp4'%i))[0]))
# video = pims.Video(str(list(p.rglob('exp_transport_stage_%i/edgertronic/*.mov'%i))[j]))
img = (sk.color.rgb2gray(video[4000]) * 255).astype(np.uint8)
img = cv.medianBlur(img,3)
hough(img)

# edgertronic_params = {
# 1: [],
# 2: [],
# 3: [24, 20, 55],
# 4: [],
# 5: [26, 20, 55],
# 6: [25, 20, 55],
# 7: [],
# 8: [24, 20, 55],
# 9: [],
# 10: [24, 15, 55],
# 11: [24, 15, 55],
# 12: [],
# }

# manta_params = {
# 1: [14, 50, 50],
# 2: [14, 50, 50],
# 3: [14, 50, 55],
# 4: [14, 50, 55],
# 5: [14, 50, 60],
# 6: [14, 60, 60],
# 7: [14, 55, 55],
# 8: [14, 60, 60],
# 9: [14, 55, 55],
# 10: [],
# 11: [14, 55, 55],
# 12: [14, 55, 55],
# }
