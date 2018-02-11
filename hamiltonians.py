# -*- coding: utf-8 -*-
""" helper functions for tf, to create hamiltonians and calculate equations of motion """
import tensorflow as tf

# hamiltonian function for double pendulum
def double_pendulum_hamiltonian(l1,l2,m1,m2,g,q,p):
    with tf.variable_scope('hamiltonian'):
        
        H = (l2**2*m2*p[:,0]**2 + l1**2*(m1+m2)*p[:,1]**2         \
             - 2*m2*l1*l2*p[:,0]*p[:,1]*tf.cos(q[:,0]-q[:,1]))      \
        / (2*l1**2*l2**2*m2*(m1 + m2*(tf.sin(q[:,0]-q[:,1]))**2)) \
        - m2*g*l2*tf.cos(q[:,1]) - (m1 + m2)*g*l1*tf.cos(q[:,0])        
        
        with tf.variable_scope('summary'):
            tf.summary.scalar('energy', H)
        return H

# hamiltonian function for string modeled by series of masses and springs
def string_hamiltonian(qval,p):
    with tf.variable_scope('hamiltonian'):
        with tf.variable_scope('kinetic_energy'):
            T = tf.reduce_sum(1/2 * p**2)
        with tf.variable_scope('potential_energy'):
            V = qval[:,0]**2/2 + qval[:,-1]**2/2
            V = V + tf.reduce_sum(tf.nn.conv1d(tf.expand_dims(qval,2),filters=tf.reshape([1.,-1.],(-1,1,1)),stride=1,padding='VALID')**2/2)   
        return T + V

def split_coordinates(x):
    return tf.split(x, 2, axis=1)    
    
def stack_coordinates(q,p):
    return tf.concat([q,p],axis=2)[0]
        
# return time derivative of a state given its hamiltonian function
def hamiltonian_time_derivative(hamiltonian,t, x):
    with tf.variable_scope('time_derivative'):
        
        with tf.variable_scope('split_position_momentum'):
            qval,pval = split_coordinates(x)
    
        H = tf.reduce_sum(hamiltonian(qval,pval))
            
        with tf.variable_scope('cannonical_equations'):
            dpdt = tf.gradients(-H,qval)
            dqdt = tf.gradients(H,pval)
    
        with tf.variable_scope('recombine'):
            dxdt = stack_coordinates(dqdt,dpdt)
            
    return dxdt

