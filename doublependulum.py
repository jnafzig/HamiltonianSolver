# =============================================================================
# # -*- coding: utf-8 -*-
# from functools import partial
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import time
# 
# from hamiltonians import double_pendulum_hamiltonian, hamiltonian_time_derivative
# from solver import rk4_step, update_state
# 
# tf.reset_default_graph()
# sess = tf.InteractiveSession()
# 
# # setup simulation
# l1 = 1
# l2 = 1
# m1 = 1
# m2 = 1
# g = 10
# 
# height = 10;
# width = 2*(height-1)
# theta1 = np.linspace(0,np.pi,num=height,endpoint=True)
# theta2 = np.linspace(0,2*np.pi,num=width,endpoint=False)
# # =============================================================================
# # width = 1900
# # height = 300
# # theta1 = np.linspace(3.75,4.6,num=height)
# # theta2 = np.linspace(0,np.pi,num=width)
# # =============================================================================
# 
# t1,t2 = np.meshgrid(theta1,theta2)
# 
# initial_angles = np.hstack((np.reshape(t1,(-1,1)),np.reshape(t2,(-1,1))))
# initial_momenta = np.zeros(initial_angles.shape)
# 
# h = tf.constant(.01, dtype=tf.float32, name='time_step');
# #x0 = [[np.random.rand()*np.pi,2*np.random.rand()*np.pi,0,0] for i in range(360000)]
# x0 = np.hstack((initial_angles,initial_momenta))
# t0 = 0;
# 
# with tf.variable_scope('state'):
#     x = tf.Variable(x0, dtype=tf.float32)
#     t = tf.Variable(0, dtype=tf.float32)
# 
# hamiltonian = partial(double_pendulum_hamiltonian, l1,l2,m1,m2,g)
# dxdt = partial(hamiltonian_time_derivative, hamiltonian)
# dx = rk4_step(dxdt, t, x, h)
# update = update_state(x, t, dx, h)
# 
# sess.run(tf.global_variables_initializer())
# summary_writer = tf.summary.FileWriter('logdir/', sess.graph)
# 
# N = 1000
# state = np.array([x0]*N, dtype=np.float32)
# 
# # run simulation
# start = time.time()
# 
# for i in range(N):
#     _, state[i] = sess.run([update,x])
# 
# end = time.time()
# print(end - start)
# 
# 
# summary_writer.close()
# sess.close()
# =============================================================================

i = 18

x1 = l1 * np.sin(state[:,i,0])
y1 = -l1 * np.cos(state[:,i,0])
x2 = x1 + l2 * np.sin(state[:,i,1])
y2 = y1 - l2 * np.cos(state[:,i,1])

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
                              interval=10, blit=True)


plt.show()



