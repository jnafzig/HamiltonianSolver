
# -*- coding: utf-8 -*-
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pickle

from eom import double_pendulum_eom, double_pendulum_jacobian

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

height = 100
width = 100
theta1 = np.linspace(-np.pi,np.pi,num=height,endpoint=True)
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
jac = double_pendulum_jacobian(dxdt,t,x)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

min_energy = sess.run([hamiltonian],feed_dict={x: [[0,np.pi,0,0]]})[0]
energy0 = sess.run([hamiltonian],feed_dict={x: x0})[0]
mask = energy0 > min_energy
indices = np.where(mask)[0]

state = np.array(x0, dtype=np.float32)

jac_val = sess.run([jac],feed_dict={x: x0})[0]
eigvals = np.linalg.eigvals(jac_val)

plt.imshow(np.imag(eigvals[:,0].reshape((height,width))))

summary_writer.close()
sess.close()




