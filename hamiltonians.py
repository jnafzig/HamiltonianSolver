# -*- coding: utf-8 -*-
import tensorflow as tf

# hamiltonian function for double pendulum
def double_pendulum_hamiltonian(l1,l2,m1,m2,g,q,p):
    with tf.variable_scope('hamiltonian'):
        #q1, q2 = tf.unstack(q)
        #p1, p2 = tf.unstack(p)
        
        H = (l2**2*m2*p[0]**2 + l1**2*(m1+m2)*p[1]**2 - 2*m2*l1*l2*p[0]*p[1]*tf.cos(q[0]-q[1])) / (2*l1**2*l2**2*m2*(m1 + m2*(tf.sin(q[0]-q[1]))**2)) - m2*g*l2*tf.cos(q[1]) - (m1 + m2)*g*l1*tf.cos(q[0])
        
        
        with tf.variable_scope('summary'):
            tf.summary.scalar('energy', H)
        return H

# hamiltonian function for string modeled by series of masses and springs
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
    
# return time derivative of a state given its hamiltonian function
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