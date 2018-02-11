# -*- coding: utf-8 -*-
""" Plot plot grid of pendulums where each pixel represents a pendulum. 
Run doublependulumeom.py or doublependulum.py to generate pickle of data """
import pickle
from plot import plot_pendulum_pixels

state, setup = pickle.load( open( "pend_data_eom.p", "rb" ) )

ani = plot_pendulum_pixels(state,setup,mirror=False)


