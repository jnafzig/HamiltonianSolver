# -*- coding: utf-8 -*-
""" functions for setting up numerical integration """

import tensorflow as tf

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
    
    with tf.variable_scope('new_state'):
        return h/6 * (k1 + k2*2 + k3*2 + k4)

def update_state(dx,h,x,t,step):
    with tf.variable_scope('update_state'):
        return [tf.add(x, dx),
            tf.add(t, h),tf.add(step,1)]
