# -*- coding: utf-8 -*-
""" Plot single pendulum. 
Run doublependulumeom.py or doublependulum.py to generate pickle of data """
import pickle
from plot import plot_double_pendulum

state, setup = pickle.load( open( "pend_data_eom.p", "rb" ) )

ani = plot_double_pendulum(state,setup,29)


