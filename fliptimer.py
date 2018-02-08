
# -*- coding: utf-8 -*-
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pickle

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

height = 1000;
width = 2*(height-1)
theta1 = np.linspace(0,np.pi,num=height,endpoint=True)
theta2 = np.linspace(-np.pi,np.pi,num=width,endpoint=False)

t1,t2 = np.meshgrid(theta1,theta2)

initial_angles = np.hstack((np.reshape(t1,(-1,1)),np.reshape(t2,(-1,1))))
initial_momenta = np.zeros(initial_angles.shape)

h = tf.constant(dt, dtype=tf.float64, name='time_step');
x0 = np.hstack((initial_angles,initial_momenta))
t0 = 0;

with tf.variable_scope('state'):
#    x = tf.Variable(x0, dtype=tf.float32)
    x = tf.placeholder(tf.float64,(None,4), name="state_input")
    t = tf.Variable(t0, dtype=tf.float64)

q, p = split_coordinates(x)
hamiltonian = double_pendulum_hamiltonian(l1,l2,m1,m2,g,q,p)

dxdt = partial(double_pendulum_eom, l1,l2,m1,m2,g)
dx = rk4_step(dxdt, t, x, h)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

min_energy = sess.run([hamiltonian],feed_dict={x: [[0,np.pi,0,0]]})[0]
energy0 = sess.run([hamiltonian],feed_dict={x: x0})[0]
mask = energy0 > min_energy
indices = np.where(mask)[0]

state = np.array(x0, dtype=np.float32)

# run simulation
start = time.time()
mid = start
i = 0
flipped = np.zeros(state[:,0].shape)
while indices.shape[0]>0:
    i = i + 1
    dstate = sess.run([dx],feed_dict={x: state[indices]})
    state[indices] = state[indices] + dstate
    
    new_flips = (state[indices][:,1]>np.pi) | (state[indices][:,1]<-np.pi)
    flipped[indices[new_flips]] = i*dt
    mask[indices[new_flips]] = False
    indices = indices[np.logical_not(new_flips)]
    
# =============================================================================
#     flipped[np.logical_not(flipped) & ((state[:,1]>np.pi) | (state[:,1]<-np.pi))] = i*dt
#     mask = mask & np.logical_not(flipped)
# =============================================================================
    if i % 100 == 0:
        now = time.time()
        print(now-mid,np.sum(mask)/mask.shape[0],i, i*dt, now-start)
        mid = now
        
    if i % 1000 == 0:
        flipped_image = flipped.reshape(t1.shape)
        pickle.dump( flipped_image, open( "flipped.p", "wb" ) )

    
end = time.time()
    
print(end - start)

summary_writer.close()
sess.close()




