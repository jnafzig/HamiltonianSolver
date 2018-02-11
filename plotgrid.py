# -*- coding: utf-8 -*-
""" Plot grid of pendulums. 
Run doublependulumeom.py or doublependulum.py to generate pickle of data """
import pickle
from plot import plot_mini_pendula

state, setup = pickle.load( open( "pend_data_eom.p", "rb" ) )

ani = plot_mini_pendula(state,setup)

