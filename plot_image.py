# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pylab import *
import matplotlib.cm as cm
from matplotlib import animation


from eom import double_pendulum_eom

from hamiltonians import double_pendulum_hamiltonian, hamiltonian_time_derivative, split_coordinates
from solver import rk4_step, update_state

tf.reset_default_graph()
sess = tf.InteractiveSession()

# setup simulation
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 10
dt = 0.01

h = tf.constant(dt, dtype=tf.float64, name='time_step')
x0 = np.array([[1,-1,0,0]])
t0 = 0

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

new_coordinate_gen = partial(integrate,l1,l2,sess,dx)

coordinate_gen = new_coordinate_gen(x0)

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


ax2.imshow(np.log10(flipped), extent=(0,np.pi,np.pi,-np.pi), axes=ax2)

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

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.ax2: return
        if not event.dblclick: return
        
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
# =============================================================================
# image = np.log10(np.flipud(flipped.T))
# image = color_map(image)
# 
# fig = plt.figure(figsize=(20,10))
# ax = Axes(fig,[0,0,1,1],yticks=[],xticks=[],frame_on=False)
# plt.gcf().delaxes(plt.gca())
# plt.gcf().add_axes(ax)
# im = plt.imshow(image,interpolation='bicubic',aspect='auto')
# 
# # Now adding the colorbar
# cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
# cb = plt.colorbar(im, cax = cbaxes,orientation="horizontal")  
# cb.set_ticks([0,1])
# cb.set_ticklabels([-1.0,4.059541456509082],update_ticks=True)
# =============================================================================
# =============================================================================
# plt.show()
# 
# savefig('demo.png', transparent=True)
# =============================================================================
