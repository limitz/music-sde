# Synthesize Music using Stochastic Differential Equations.

In this experiment I'm going to use SDE's to generate music (hopefully). The idea behind these kinds of networks is that you have a forward diffusion process that transforms a sample `S` into normally distributed noise `N(0,1)`, simply by linearly interpolating the `mean` from `S` to 0 and the `std` from 0 to 1 over time and then sampling from `N(mean, std)`. This is the forward stochastic differential equation:
```
dx = f(x, t)dt + g(t)dw, 
```
where `f` is the drift coefficient (or mean over time) and `g` is the diffusion coefficient (variance over time) and `w` is a wiener process (randomness).

This diffusion process can be reversed by using a reverse time SDE. 
```
dx = [f (x, t) − g(t)^2∇x log pt(x)]dt + g(t)d  ̄w
```

This SDE will transform the `N(0,1)` noise back to it's original state, iteratively, **if** you provide the _score_ `∇x log pt(x)` at each time step. This _score_ is unknown during inference, but we can train a model to predict this `score` from the output image at each time step.



### The model is an adaptation of an example in the [torchsde repo](https://github.com/google-research/torchsde/blob/master/examples/cont_ddpm.py).

The Unet implementation is changed to a 1d Unet, and has some other changes done during refactoring because the example code was a bit convoluted to me. I always try to reinterpret code and write it down from there when I'm learning about something new instead of copy pasting. That way the code becomes my own and I understand it better.
