
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
dt = 0.02

height = 30
width = 30
theta1 = np.linspace(-np.pi,np.pi,num=height)
theta2 = np.linspace(-np.pi,np.pi,num=width)
thetadot1 = np.linspace(0,35,num=height)
thetadot2 = np.linspace(0,25,num=width)
# =============================================================================
# width = 1900
# height = 300
# theta1 = np.linspace(3.75,4.6,num=height)
# theta2 = np.linspace(0,np.pi,num=width)
# =============================================================================

t1,t2 = np.meshgrid(theta1,theta2)
td1,td2 = np.meshgrid(thetadot1,thetadot2)

# =============================================================================
# init_td1 = np.reshape(td1,(-1,1))
# init_td2 = np.reshape(td2,(-1,1))
# init_t1 = np.zeros(init_td1.shape)
# init_t2 = np.zeros(init_td2.shape)
# =============================================================================

init_t1 = np.reshape(t1,(-1,1))
init_t2 = np.reshape(t2,(-1,1))
init_td1 = np.zeros(init_t1.shape)
init_td2 = np.zeros(init_t2.shape)

h = tf.constant(dt, dtype=tf.float32, name='time_step');
#x0 = [[np.random.rand()*np.pi,2*np.random.rand()*np.pi,0,0] for i in range(360000)]
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

N = 1000
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






