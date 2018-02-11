# -*- coding: utf-8 -*-
""" Make a large plot of the time to flip data"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import pickle
import matplotlib.cm as cm

def color_map(image):
    im = np.zeros((image.shape[0],image.shape[1],4))
    mask = image == -np.inf
    fill_color = np.array(cm.viridis(0))
    fill_color[3] = 0
    im[mask] = fill_color
    minval = np.min(image[np.logical_not(mask)])
    maxval = np.max(image[np.logical_not(mask)])
    im[np.logical_not(mask)] = cm.viridis((image[np.logical_not(mask)]-minval)/(maxval-minval))
    return im

flipped = pickle.load( open( "flipped.p", "rb" ) )

image = np.log10(np.flipud(flipped.T))
image = color_map(image)

fig = plt.figure(figsize=(20,10))
ax = axs.Axes(fig,[0,0,1,1],yticks=[],xticks=[],frame_on=False)
plt.gcf().delaxes(plt.gca())
plt.gcf().add_axes(ax)
im = plt.imshow(image,interpolation='bicubic',aspect='auto')

