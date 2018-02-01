---
layout: default
title: Hamiltonian Solver
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

This code uses TensorFlow to simulate a physical system.  The novel thing about the code is that the only physics you need to specify is the [Hamiltonian][hamiltonian-wiki].  The rest is automatically determined by [automatic differentiation][autodiff-wiki].  In other words if you can specify the Hamiltonian using canonical coordinates then the code will generate and numerically (RK4) solve the equations of motion:

$$\frac{\mathrm{d}\boldsymbol{p}}{\mathrm{d}t} = -\frac{\partial \mathcal{H}}{\partial \boldsymbol{q}}\quad,\quad
\frac{\mathrm{d}\boldsymbol{q}}{\mathrm{d}t} = +\frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}$$

Below is a simulation of a vibrating string (modeled as 100 masses connected linearly by springs) generated using the code.

<div class="myvideo">
   <video  style="display:block; width:100%; height:auto;" autoplay controls loop="loop">
       <source src="im.mp4" type="video/mp4" />
   </video>
</div>

Here is the code simulating a double pendulum:

<div class="myvideo">
   <video  style="display:block; width:100%; height:auto;" autoplay controls loop="loop">
       <source src="doublependulum.mp4" type="video/mp4" />
   </video>
</div>

Below is a visualization of the TensorFlow graph.  You can see the four boxes responsible for calculating the four k values of [fourth order Runge-Kutta method][rk4-wiki] and how they are then combined into the update_state box.

![TensorFlow Graph](graph.png)

[hamiltonian-wiki]: https://en.wikipedia.org/wiki/Hamiltonian_mechanics
[autodiff-wiki]: https://en.wikipedia.org/wiki/Automatic_differentiation
[rk4-wiki]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
