# -*- coding: utf-8 -*-
""" Script genereates html/js code for interactive plot """

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import mpld3
from mpld3 import plugins, utils
import pickle


class AnimatePendulum(plugins.PluginBase):
    """A plugin for animating a double pendulum"""
    f = open('PendulumAnimatePlugin.js','r')
    JAVASCRIPT = f.read()
    f.close()

    def __init__(self, pendulum, trajectory):

        self.dict_ = {"type": "animatependulum",
                      "idpendulum": utils.get_id(pendulum),
                      "idtrajectory": utils.get_id(trajectory)}

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

flipped = color_map(np.log10(np.flipud(pickle.load( open( "flipped.p", "rb" )) )))
flipped = np.hstack([np.fliplr(np.flipud(flipped)),flipped])


fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(5,10))

im = ax2.imshow(flipped, extent=(-np.pi,np.pi,-np.pi,np.pi),interpolation='nearest')
ax2.set_title('Click to start new simulation.')
ax2.set_xlabel('Initial Angle 1 (radians)')
ax2.set_ylabel('Initial Angle 2 (radians)')

pendulum = ax1.plot([0,0,0], [0,0,0], lw=3, alpha=0.5)
trajectory = ax1.plot([], [], lw=3, alpha=0.5)

ax1.set_ylim(-2.5, 2.5)
ax1.set_xlim(-2.5, 2.5)

plugins.connect(fig, AnimatePendulum(pendulum[0],trajectory[0]))

mpld3.show()
# =============================================================================
# f = open('plot.html', 'w')
# mpld3.save_html(fig,f)
# f.close()
# =============================================================================
