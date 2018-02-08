# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm


import mpld3
from mpld3 import plugins

# plot
fig, (ax1, ax2) = plt.subplots(1, 2)

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


ax2.imshow(np.log10(flipped), extent=(0,np.pi,-np.pi,np.pi), axes=ax2,origin='lower', zorder=1, interpolation='nearest')

ax2.set_title('simulation time till second pendulum flips')
ax2.set_xlabel('Initial Angle 1 (radians)')
ax2.set_ylabel('Initial Angle 2 (radians)')


plugins.connect(fig, plugins.MousePosition(fontsize=14))

mpld3.show()
