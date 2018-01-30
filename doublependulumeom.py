# -*- coding: utf-8 -*-
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from eom import double_pendulum_eom

from hamiltonians import double_pendulum_hamiltonian, hamiltonian_time_derivative
from solver import rk4_step

tf.reset_default_graph()
sess = tf.InteractiveSession()

# setup simulation
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 10

height = 10;
width = 2*(height-1)
theta1 = np.linspace(0,np.pi,num=height,endpoint=True)
theta2 = np.linspace(0,2*np.pi,num=width,endpoint=False)
# =============================================================================
# width = 1900
# height = 300
# theta1 = np.linspace(3.75,4.6,num=height)
# theta2 = np.linspace(0,np.pi,num=width)
# =============================================================================

t1,t2 = np.meshgrid(theta1,theta2)

initial_angles = np.hstack((np.reshape(t1,(-1,1)),np.reshape(t2,(-1,1))))
initial_momenta = np.zeros(initial_angles.shape)

h = tf.constant(.03, dtype=tf.float32, name='time_step');
#x0 = [[np.random.rand()*np.pi,2*np.random.rand()*np.pi,0,0] for i in range(360000)]
x0 = np.hstack((initial_angles,initial_momenta))
t0 = 0;

with tf.variable_scope('state'):
    x = tf.Variable(x0, dtype=tf.float32)
    t = tf.Variable(t0, dtype=tf.float32)

dxdt = partial(double_pendulum_eom, l1,l2,m1,m2,g)
update = rk4_step(dxdt, t, x, h)



sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

N = 400
state = np.array([x0]*N, dtype=np.float32)

# run simulation
start = time.time()

for i in range(N):
    _, state[i] = sess.run([update,x])

end = time.time()
print(end - start)

summary_writer.close()
sess.close()

# =============================================================================
# from plot import plot_mini_pendula
# ani = plot_mini_pendula(state,t1,t2,l1,l2)
# plt.show()
# =============================================================================

# =============================================================================
# from plot import plot_double_pendulum
# ani = plot_double_pendulum(state,l1,l2,5)
# plt.show()
# =============================================================================

# =============================================================================
# from plot import plot_image
# ani = plot_image(state,t1,t2)
# plt.show()
# =============================================================================




# =============================================================================
# flipped1 = (state[-1,:,0]>np.pi) | (state[-1,:,0]<-np.pi)
# flipped2 = (state[-1,:,1]>np.pi) | (state[-1,:,1]<-np.pi)
# maxval0 = max(state[-1,:,0])
# minval0 = min(state[-1,:,0])
# maxval1 = max(state[-1,:,1])
# minval1 = min(state[-1,:,1])
# color = np.vstack([flipped1,flipped2,1-flipped2]).T
# 
# plt.scatter(state[0,:,0],state[0,:,1],c=color,marker=',',s=1)
# =============================================================================



