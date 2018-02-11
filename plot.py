# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_mini_pendula(state,setup,save=False):
    """ Plot grid of pendulums """
    
    height_range = setup['height_range']
    width_range = setup['width_range']
    
    height_spacing = height_range[1] - height_range[0]
    width_spacing = width_range[1] - width_range[0]
    
    grid1, grid2 = np.meshgrid(height_range,width_range)
    
    l1 = setup['l1']
    l2 = setup['l2']
    
    state = np.array(state)
    N = state.shape[0]
    
    x1 = l1 * np.sin(state[:,:,0])
    y1 = -l1 * np.cos(state[:,:,0])
    x2 = x1 + l2 * np.sin(state[:,:,1]) 
    y2 = y1 - l2 * np.cos(state[:,:,1]) 
    
    xvals = np.stack([0.0*x1,x1,x2],axis=1)*height_spacing/2 + grid1.reshape((1,1,-1))
    yvals = np.stack([0.0*y1,y1,y2],axis=1)*width_spacing/2 + grid2.reshape((1,1,-1))
    
    # plot
    fig, ax = plt.subplots(figsize=(7, 7))
    
    pendulums = ax.plot(xvals[0], yvals[0], color='k', linestyle='-', linewidth=1)
    
    ax.axis('equal')
    
    plt.title('Double Pendula with various initial conditions')
    plt.ylabel('Angle 1 (radians)')
    plt.xlabel('Angle 2 (radians)')
    
    
    def animate(i):
        for j, pendulum in enumerate(pendulums):
            pendulum.set_data(xvals[i,:,j],yvals[i,:,j])
            
        return pendulums
    
    ani = animation.FuncAnimation(fig, animate, range(N),
                                  interval=15, blit=True)
    
    plt.show()
    
    if save:
        #Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'))
        
        ani.save('minipendulum.mp4', writer=writer)


    return ani

def plot_double_pendulum(state,setup,i,save=False):
    """ Plot single pendulum plus trajectory"""
    
    l1 = setup['l1']
    l2 = setup['l2']
    
    state = np.array(state)
    N = state.shape[0] 
    
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
                                  interval=25, blit=True)
    
    if save:
        #Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        
        ani.save('doublependulum.mp4', writer=writer)
    
    plt.show()
    
    return ani


def plot_pendulum_pixels(state,setup,save=False,mirror=True):
    
    
    height_range = setup['height_range']
    width_range = setup['width_range']
    
    grid1, grid2 = np.meshgrid(height_range,width_range)
    
    state = np.array(state)
    N = state.shape[0]
    
    angle1 = np.reshape(state[0,:,0],grid1.shape)
    angle2 = np.reshape(state[0,:,1],grid2.shape)
    if mirror:
        angle1 = np.hstack((angle1,np.flipud(np.fliplr(-angle1[:,:-1]))))
        angle2 = np.hstack((angle2,np.flipud(np.fliplr(-angle2[:,:-1]))))
    
    fig = plt.figure(figsize=(8,6))
    
# =============================================================================
#     im = plt.imshow(np.sin(angle1), extent=(np.min(t1),2*np.max(t1),np.min(t2),np.max(t2)))
# =============================================================================
    
    im = plt.imshow(np.sin(angle1),
                    extent=(np.min(grid1),2*np.max(grid1),np.min(grid2),np.max(grid2)),
                    vmin=-1,vmax=1,
                    interpolation='bicubic')
    
    plt.colorbar()
    plt.title('sin of Angle 1')
    
    def updatefig(i):
        angle1 = np.reshape(state[i,:,0],grid1.shape)
        angle2 = np.reshape(state[i,:,1],grid2.shape)
        if mirror:
            angle1 = np.hstack((angle1,np.flipud(np.fliplr(-angle1[:,:-1]))))
            angle2 = np.hstack((angle2,np.flipud(np.fliplr(-angle2[:,:-1]))))
        
        im.set_array(np.sin(angle1))
        return im,
    
    ani = animation.FuncAnimation(fig, updatefig,range(N), interval=50, blit=True, repeat=True)
    plt.show()
    
    if save:
        #Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        
        ani.save('sin_angle_1_zoom_full.mp4', writer=writer)

    return ani

