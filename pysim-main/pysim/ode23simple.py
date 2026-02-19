import sys
import numpy as np
from collections import namedtuple
from numpy.linalg import norm
from .FastSimODE import FastSimODE

def ode23simple(TOUT,T,x0,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu):
    # Simplified version of Matlab's ode23
    #
    # Usage: [tout,xout,stats] = ode23simple(TOUT,x0)
    #
    # Integrates the ODE system dx/dt = f(t,x) from t=t0 to t=T
    #   with initial condition x(t0)=x0
    #   using a 3-stage Runge-Kutta method (due to Bogacki and Shampine)
    #   with adaptive step size h
    #
    # TOUT  desired output times
    # x0    initial state vector
    # tol   tolerance (see above)
    #
    # tout  output times
    # xout  output states
    # stats debugging statistics
    #
    # REFERENCE
    # see (Apr 13,15) Numerical solution of ODEs: adaptive algorithms and Matlab's ODE solvers [PDF]; ode23smp.m
    # at http://www.dam.brown.edu/people/alcyew/apma0160.html
    #
    # H. Kirk Mathews (GE Research) 18 July 2018
    # Joe Wakeman     (Wabtec)      31 July 2024 - Translated to Python

    # default tolerances
    rtol = 1.e-2
    atol = 1.e-3

    threshold = atol/rtol

    # initialize
    t0 = TOUT[0]
    tEnd = TOUT[-1]
    hmax = 0.1*(tEnd-t0)
    t = t0
    x = np.copy(x0)
    tout = np.copy(TOUT)
    xout = np.zeros((np.size(TOUT), np.size(x)))
    tout[0]   = t
    xout[0,:] = x.transpose()
    stats = namedtuple('stats', ['err', 'hmax', 'hmin', 'nsteps', 'nfailed'])
    stats.err = 0

    # choose an initial step size based on the overall scale of the problem
    s1, Frope, F, Fmin, Fmax = FastSimODE(t,x,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu) # calculate the first slope
    r = norm(np.divide(s1,np.clip(abs(x),threshold,None)),np.inf) + np.finfo(float).tiny
    h = 0.8*rtol**(1/3)/r

    stats.hmax = h
    stats.hmin = h
    stats.nsteps  = 0
    stats.nfailed = 0

    knext = 1
    nout  = np.size(TOUT)-1
    tnext = TOUT[knext]

    # the time-stepping loop
    while t < tEnd:
        
        hmin = 16*sys.float_info.epsilon*abs(t)
        
        if h > hmax:
            h = hmax
        if h < hmin:
            h = hmin

        if 1.1*h >= tnext - t: # stretch the step if t is close to but less than tnext, shrink it if beyond
            h = tnext - t
            attout = True
        else:
            attout = False

        # take a step (s1 was already calculated above)
        s2, Frope, F, Fmin, Fmax = FastSimODE(t+h/2,x+h/2*s1,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu)
        s3, Frope, F, Fmin, Fmax = FastSimODE(t+3*h/4,x+3*h/4*s2,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu)

        tnew = t + h
        xnew = x + h*(2*s1 + 3*s2 + 4*s3)/9

        # estimate the error
        s4, Frope, F, Fmin, Fmax = FastSimODE(tnew,xnew,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu)
        e = h*(-5*s1 + 6*s2 + 8*s3 - 9*s4)/72

        err_est = norm(np.divide(e,np.clip(np.maximum(abs(x),abs(xnew)),threshold,None)),np.inf) + np.finfo(float).tiny

        # accept the solution if the estimated error is less than the tolerance
        if err_est <= rtol:
            t = tnew
            x = xnew
            stats.hmax = max(stats.hmax,h)
            stats.hmin = min(stats.hmin,h)
            stats.nsteps = stats.nsteps+1
            if attout:
                tout[knext] = t
                xout[knext,:] = x.transpose()
                knext = knext+1
                if knext>nout:
                    knext = nout
                
                tnext = TOUT[knext]
            
            s1 = s4
        else:
            stats.nfailed = stats.nfailed+1
        
        # find a new step size
        h = h*min(5, 0.8*(rtol/err_est)**(1/3))
        
        # exit early if step size is too small
        if h <= hmin:
            stats.err = 1 # sprintf('Step size %e too small at t = %e.\n',h,t);
            break
    
    return tout,xout,stats
