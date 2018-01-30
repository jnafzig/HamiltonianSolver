# -*- coding: utf-8 -*-
import tensorflow as tf

def double_pendulum_eom(l1,l2,m1,m2,g,t,x):
    with tf.variable_scope('eom'):
              
        q1,q2,p1,p2 = tf.unstack(x, axis=1)
        
        qdiff = q1-q2
        cosdiff = tf.cos(qdiff)
        sindiff = tf.sin(qdiff)
        qdenom = m1 + m2*sindiff**2
        qdot1 = (l2*p1 - l1*p2*cosdiff)/(l1**2*l2*qdenom)
        qdot2 = (l1*(m1+m2)*p2 - l2*m2*p1*cosdiff)/(l1*l2**2*qdenom)
        c1 = p1*p2*sindiff/(l1*l2*qdenom)
        c2 = (l2**2*m2*p1**2 + l1**2*(m1+m2)*p2**2 - 2*l1*l2*m2*p1*p2*cosdiff)*tf.sin(2*qdiff)/(2*l1**2*l2**2*qdenom**2)
        pdot1 = -(m1+m2)*g*l1*tf.sin(q1) - c1 + c2
        pdot2 = -m2*g*l2*tf.sin(q2) + c1 - c2
        
        dxdt = tf.stack([qdot1,qdot2,pdot1,pdot2], axis=1)
        
        return dxdt