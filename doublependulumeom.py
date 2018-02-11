# -*- coding: utf-8 -*-
""" Runs N = height * width pendulum simulations and saves the data.  Directly
calculates the equations of motion"""

from functools import partial
import tensorflow as tf
import numpy as np
import time
import pickle

from eom import double_pendulum_eom

from hamiltonians import double_pendulum_hamiltonian, split_coordinates
from solver import rk4_step, update_state

# setup simulation parameters
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 10
dt = 0.005

height = 30
width = 30

#Form grid of regularly spaced angles for initial conditions:

# =============================================================================
# theta1 = np.linspace(-np.pi,np.pi,num=height)
# theta2 = np.linspace(-np.pi,np.pi,num=width)
# 
# dtheta1 = theta1[1]-theta1[0]
# dtheta2 = theta2[1]-theta2[0]
# 
# t1,t2 = np.meshgrid(theta1,theta2)
# 
# init_t1 = np.reshape(t1,(-1,1))
# init_t2 = np.reshape(t2,(-1,1))
# init_td1 = np.zeros(init_t1.shape)
# init_td2 = np.zeros(init_t2.shape)
# =============================================================================

#Form grid of regularly spaced momenta for initial conditions:

thetadot1 = np.linspace(0,35,num=height)
thetadot2 = np.linspace(0,25,num=width)

td1,td2 = np.meshgrid(thetadot1,thetadot2)

init_td1 = np.reshape(td1,(-1,1))
init_td2 = np.reshape(td2,(-1,1))
init_t1 = np.zeros(init_td1.shape)
init_t2 = np.zeros(init_td2.shape)

setup = dict(l1=l1,l2=l2,m1=m1,m2=m2,g=g,
             height_range=thetadot1,
             width_range=thetadot2,
             dt=dt)

# setup computational graph

tf.reset_default_graph()
sess = tf.InteractiveSession()

h = tf.constant(dt, dtype=tf.float32, name='time_step');
x0 = np.hstack((init_t1,init_t2,init_td1,init_td2))
t0 = 0;

with tf.variable_scope('state'):
    x = tf.Variable(x0, dtype=tf.float32)
    t = tf.Variable(t0, dtype=tf.float32)

q, p = split_coordinates(x)
hamiltonian = double_pendulum_hamiltonian(l1,l2,m1,m2,g,q,p)

dxdt = partial(double_pendulum_eom, l1,l2,m1,m2,g)
dx = rk4_step(dxdt, t, x, h)
update = update_state(x, t, dx, h)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

# run simulation
Numiterations = 1000
state = np.array([x0]*Numiterations, dtype=np.float32)

start = time.time()

for i in range(Numiterations):
    _, state[i] = sess.run([update,x])
    
end = time.time()
    
print(end - start)

summary_writer.close()
sess.close()

# save data
f = open("pend_data_eom.p", "wb")
pickle.dump([state.tolist(), setup], f)
f.close()



