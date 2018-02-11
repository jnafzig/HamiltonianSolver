# -*- coding: utf-8 -*-
""" Creates interactive plot using mpl """
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import animation

import sys
sys.path.insert(0, '..')
from eom import double_pendulum_eom
from solver import rk4_step

tf.reset_default_graph()
sess = tf.InteractiveSession()

# setup simulation
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 10
dt = 0.01
x0 = np.array([[1,-1,0,0]])
t0 = 0

# setup computational graph
h = tf.constant(dt, dtype=tf.float64, name='time_step')

with tf.variable_scope('state'):
    x = tf.placeholder(tf.float64,(None,4), name="state_input")
    t = tf.Variable(t0, dtype=tf.float64)

dxdt = partial(double_pendulum_eom, l1,l2,m1,m2,g)
dx = rk4_step(dxdt, t, x, h)

def integrate(l1,l2,sess,dx,x0):
    state = np.array(x0, dtype=np.float32)
    while True:
        x1 = l1 * np.sin(state[0,0])
        y1 = -l1 * np.cos(state[0,0])
        x2 = x1 + l2 * np.sin(state[0,1])
        y2 = y1 - l2 * np.cos(state[0,1])
        yield [[0,x1,x2],[0,y1,y2]]
        state = state + sess.run([dx],feed_dict={x: state})[0]

sess.run(tf.global_variables_initializer())

# create function to be passed to animate function
new_coordinate_gen = partial(integrate,l1,l2,sess,dx)
coordinate_gen = new_coordinate_gen(x0)

# setup plot
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

flipped = pickle.load( open( "flipped.p", "rb" ) )
flipped = np.hstack([np.fliplr(np.flipud(flipped)),flipped])

ax2.imshow(np.log10(flipped), extent=(-np.pi,np.pi,np.pi,-np.pi))

plt.title('simulation time till second pendulum flips')
plt.xlabel('Initial Angle 1 (radians)')
plt.ylabel('Initial Angle 2 (radians)')


class PlotConnect:
    def __init__(self, figure,ax1,ax2,new_coordinate_gen):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ani = None
        self.new_coordinate_gen = new_coordinate_gen
        self.cid = figure.canvas.mpl_connect('button_press_event', self)
        self.plot_pendulum(self.new_coordinate_gen([[1,-1,0,0]]))
        self.marker = ax2.plot(1,-1,"+")[0]

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.ax2: return
        if not event.dblclick: return

        self.marker.set_data(event.xdata,event.ydata)
        self.ax2.draw_artist(self.marker)
        ax2.get_figure().canvas.blit(ax2.bbox)
        self.ani._stop()
        self.plot_pendulum(self.new_coordinate_gen([[event.xdata,event.ydata,0,0]]))

    def plot_pendulum(self,coordinate_gen):
        trajectory_x = []
        trajectory_y = []

        def animate(coordinates):
            trajectory_x.append(coordinates[0][2])
            trajectory_y.append(coordinates[1][2])
            pendulum.set_data(coordinates[0], coordinates[1])
            trajectory.set_data(trajectory_x,trajectory_y)
            return pendulum, trajectory,

        trajectory, = self.ax1.plot(trajectory_x, trajectory_y, color='b', linestyle='-', linewidth=1)
        pendulum, = self.ax1.plot([np.nan,np.nan,np.nan], [np.nan,np.nan,np.nan], color='k', linestyle='-', linewidth=2)

        self.ax1.axis('equal')
        self.ax1.axis([-2.8, 2.8, -2.8, 2.8])
        self.ax1.axes.xaxis.set_ticklabels([])
        self.ax1.axes.yaxis.set_ticklabels([])

        self.ani = animation.FuncAnimation(fig, animate, coordinate_gen,
                                      interval=15, blit=True)

plotconnect = PlotConnect(fig,ax1,ax2,new_coordinate_gen)
plt.show()
