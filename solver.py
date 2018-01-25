# -*- coding: utf-8 -*-
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tf.reset_default_graph()
sess = tf.InteractiveSession()

string_length = 100;
h = tf.constant(.05, dtype=tf.float32, name='time_step');
x0 = np.zeros(2*string_length)
x0[-string_length:] = np.array(range(string_length))/string_length

with tf.variable_scope('state'):
    x = tf.Variable(x0, dtype=tf.float32)
    t = tf.Variable(0, dtype=tf.float32)
    
def string_hamiltonian(q,p):
    with tf.variable_scope('hamiltonian'):
        with tf.variable_scope('kinetic_energy'):
            T = tf.reduce_sum(1/2 * p**2)
        with tf.variable_scope('potential_energy'):
            V = q[0]**2/2 + q[-1]**2/2
            q = tf.reshape(q,[1,-1,1,1])
            filter = tf.constant([[[[1.]]],[[[-1.]]]], dtype=tf.float32)
            V = V + tf.reduce_sum(tf.nn.conv2d(q,filter=filter,strides=[1,1,1,1],padding='VALID')**2/2)   
        with tf.variable_scope('summary'):
            tf.summary.scalar('energy', T + V)
        return T + V
    
def time_derivative(hamiltonian,t, x):
    with tf.variable_scope('time_derivative'):
        
        with tf.variable_scope('split_position_momentum'):
            q,p = tf.split(x,2)
    
        H = hamiltonian(q,p)
            
        with tf.variable_scope('cannonical_equations'):
            dpdt = tf.gradients(-H,q)
            dqdt = tf.gradients(H,p)
    
        with tf.variable_scope('recombine'):
            dxdt = tf.squeeze(tf.concat([dqdt,dpdt],axis=1))
    return dxdt

def rk4_step(time_derivative, t, x, h):
    
    with tf.variable_scope('k1'):
        k1 = time_derivative(t, x)
    with tf.variable_scope('k2'):
        with tf.variable_scope('t1'):
            t1 = t + h/2;
        with tf.variable_scope('x1'):
            x1 = x + h*k1/2;
        k2 = time_derivative(t1, x1)
    with tf.variable_scope('k3'):
        with tf.variable_scope('t2'):
            t2 = t + h/2;
        with tf.variable_scope('x2'):
            x2 = x + h*k2/2;
        k3 = time_derivative(t2, x2)
    with tf.variable_scope('k4'):
        with tf.variable_scope('t3'):
            t3 = t + h;
        with tf.variable_scope('x3'):
            x3 = x + h*k3;
        k4 = time_derivative(t3, x3)
    
    with tf.variable_scope('update_state'):
        return tf.group(tf.assign_add(x, h/6 * (k1 + k2*2 + k3*2 + k4)),
            tf.assign_add(t, h),
            name='update_state')
        
dxdt = partial(time_derivative, string_hamiltonian)
update = rk4_step(dxdt, t, x, h)

merged = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('logdir/', sess.graph)

N = 4000
state = np.zeros((2*string_length,N))

for i in range(N):
    summary, _, state[:,i] = sess.run([merged,update,x])
    summary_writer.add_summary(summary, i)

summary_writer.close()


fig, ax = plt.subplots()

line, = ax.plot(np.full((string_length),np.nan))

axes = plt.gca()
axes.set_xlim([-1,string_length])
axes.set_ylim([-30,30])

plt.title('Vibrating String')
plt.ylabel('Vertical Displacement')
plt.xlabel('Position Along String')

skip = 15

def animate(i):
    line.set_ydata(state[:string_length,skip*i])# update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(state[:string_length,0], mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, range(N//skip),
                              interval=skip, blit=True)

#Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

ani.save('im.mp4', writer=writer)

plt.show()