# -*- coding: utf-8 -*-
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines

from hamiltonians import double_pendulum_hamiltonian, time_derivative
from solver import rk4_step



tf.reset_default_graph()
sess = tf.InteractiveSession()

# setup simulation
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 10

h = tf.constant(.03, dtype=tf.float32, name='time_step');
x0 = [2,1,0,0];

with tf.variable_scope('state'):
    x = tf.Variable(x0, dtype=tf.float32)
    t = tf.Variable(0, dtype=tf.float32)

hamiltonian = partial(double_pendulum_hamiltonian, l1,l2,m1,m2,g)
dxdt = partial(time_derivative, hamiltonian)
update = rk4_step(dxdt, t, x, h)


sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

N = 400
state = np.zeros((4,N))

# run simulation
for i in range(N):
    _, state[:,i] = sess.run([update,x])

summary_writer.close()

x1 = l1 * np.sin(state[0,:])
y1 = -l1 * np.cos(state[0,:])
x2 = x1 + l2 * np.sin(state[1,:])
y2 = y1 - l2 * np.cos(state[1,:])

# plot
fig, ax = plt.subplots()

trajectory, = plt.plot(x2[:0], y2[:0], color='b', linestyle='-', linewidth=1)
pendulum, = plt.plot([np.nan,np.nan,np.nan], [np.nan,np.nan,np.nan], color='k', linestyle='-', linewidth=2)

axes = plt.gca()
axes.axis('equal')
axes.axis([-2.8, 2.8, -2.8, 2.8])
axes.axes.xaxis.set_ticklabels([])
axes.axes.yaxis.set_ticklabels([])

def animate(i):
    trajectory.set_data([x2[:i],y2[:i]])
    pendulum.set_data([0,x1[i],x2[i]], [0,y1[i],y2[i]])
    return pendulum, trajectory,

ani = animation.FuncAnimation(fig, animate, range(N),
                              interval=25, blit=True)

plt.show()

sess.close()

