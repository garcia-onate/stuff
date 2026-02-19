import numpy as np
from .ssCouplerDisplacement import ssCouplerDisplacement
from .FastSimODE import FastSimODE
from .ode23simple import ode23simple

def FastSimM(T, Frope, tstart, tend, n, b, c, M, x2all, Fmax2all, Fmin2all, Klocked, Kawu, X0='steady-state'):
    # Simulate FastSim (M file version) using the ode23simple variable step solver.
    #
    # Usage: [tode,Xode,stats] = FastSimM(T,Frope,tstart,tend,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu,X0)
    #
    # INPUTS
    # T         vector of Time associate with Frope
    # Frope     Rope forces vs time (npts x ncouplers)
    # tstart    simulation start time
    # tend      simulation end time
    # n         number of couplers
    # b         viscous damping for each coupler
    # c         non-linear damping (EOCC) for each coupler
    # M         vector of mass of cars
    # x2all     vector of coupler displacements (common for all couplers) - (nc x 1)
    # Fmax2all  hysterisis maximum coupler force associated with displacement for each coupler - (nc x ncouplers)
    # Fmin2all  hysterisis minimum coupler force associated with displacement for each coupler - (nc x ncouplers)
    # Klocked   vector of locked (between min & max hysterisis limits) coupler spring constant
    # Kawu      anti-windup gain (hysterisis modelled as saturation with anti-windup)
    # X0        Initial states ([F;DV;DX] or options, 'zero', 'steady-state')
    #           scalar value is expended, default - 'steady-state'
    #
    # OUTPUTS
    # tode      output vector of time
    # Xode      simulated output states [F1; DV; DX] for each output time (ntime x ncouplers)
    # stats     ode23simple statistics
    #
    # H. Kirk Mathews
    # GE Global Research, 2018

    # ode23 solver options
    options = {'RelTol': 1e-2, 'AbsTol': 1e-3, 'MaxStep': 20, 'NormControl': 'on'}

    # define uniform sampled displacement for force-displacement lookup table (improves speed)
    x2all_uniform = np.arange(np.floor(x2all[0]), np.ceil(x2all[-1]) + 0.01, 0.01)
    
    # Interpolate each coupler separately since np.interp doesn't handle 2D arrays directly
    Fmax2all_uniform = np.zeros((len(x2all_uniform), n))
    Fmin2all_uniform = np.zeros((len(x2all_uniform), n))
    
    for i in range(n):
        Fmax2all_uniform[:, i] = np.interp(x2all_uniform, x2all, Fmax2all[:, i])
        Fmin2all_uniform[:, i] = np.interp(x2all_uniform, x2all, Fmin2all[:, i])

    # find the start (k1) and end (k2) indices of time to simulate
    tmp = np.where(T >= tstart)[0]
    k1 = tmp[0]
    tmp = np.where(T <= tend)[0]
    k2 = tmp[-1]

    # Scale the inputs
    Fscale = 1  # not needed so set to 1
    Frope1 = Frope / Fscale
    b1 = b.flatten() / Fscale
    M1 = M.flatten() / Fscale
    Fmax2all_uniform1 = Fmax2all_uniform / Fscale
    Fmin2all_uniform1 = Fmin2all_uniform / Fscale
    Klocked1 = Klocked.flatten() / Fscale

    # default initial values
    # (X0 parameter already has default value in function signature)

    # intialize the states
    if isinstance(X0, str):
        if X0.lower() == 'zero':
            X0 = np.zeros(3 * n)
        elif X0.lower() == 'steady-state':
            # find the state-state displacement given the initial rope forces
            F10 = Frope[0, :]
            DV0 = F10 * 0
            DX0 = ssCouplerDisplacement(x2all, Fmin2all, Fmax2all, Frope)
            X0 = np.concatenate([F10, DV0, DX0])
            X0[np.isnan(X0)] = 0  # default to zero if NaN encountered
        else:
            raise ValueError(f'FastSimM: invalid initial state option, "{X0}"')
    else:
        # numerical initial conditions - apply scaler expansion if needed
        if np.isscalar(X0) or len(X0) == 1:
            X0 = np.zeros(3 * n) + X0

    # solve ODE using adaptive stepsize solver, ode23 (3-stage Runge-Kutta method, due to Bogacki and Shampine)
    tode, Xode, stats = ode23simple(T[k1:k2+1], T, X0.flatten(), Frope1, n, b1, c.flatten(), M1, x2all_uniform, Fmax2all_uniform1, Fmin2all_uniform1, Klocked1, Kawu.flatten())

    # unscale outputs
    Xode[:, 0:n] = Xode[:, 0:n] * Fscale

    # Call the ODE model to get derivatives and outputs at solved states
    Xdot = np.zeros(Xode.shape)           # state derivatives
    F = np.zeros((Xode.shape[0], n))      # coupler forces
    Fmin = np.zeros((Xode.shape[0], n))   # min of force-displacement hystersis curve at current displacement
    Fmax = np.zeros((Xode.shape[0], n))   # max of force-displacement hystersis curve at current displacement
    for i in range(Xode.shape[0]):
        Xdot[i, :], _, F[i, :], Fmin[i, :], Fmax[i, :] = FastSimODE(tode[i], Xode[i, :], T, Frope1, n, b1, c.flatten(), M1, x2all_uniform, Fmax2all_uniform1, Fmin2all_uniform1, Klocked1, Kawu.flatten())
    Yode = np.column_stack([F, Fmin, Fmax, Xdot])

    return tode, Xode, stats, Yode