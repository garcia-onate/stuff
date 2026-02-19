import numpy as np
import math

def FastSimODE(t,X,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu):
    # Compute the derivatives from the ODE of the FastSim in-trian force model.
    #
    # Usage: [Xdot,Frope,F,Fmin,Fmax] = FastSimODE(t,X,T,FROPE,n,b,c,M,x2all,Fmax2all,Fmin2all,Klocked,Kawu)
    #
    # t         = time
    # X         = state vector, X(1:n)     = F1 (force before limits)
    #                           X(n+1:2n)  = DV (relative velocity between cars)
    #                           X(2n+1:3n) = DX (relative distance between cars, i.e. displacement)
    # T         = Time vector associated with FROPE (assumed fixed step size)
    # FROPE     = Rope forces vs time (npts x ncouplers)
    # n         = number of vehicles in the train, cars + locos
    # b         = viscous damping for each coupler
    # c         = non-linear damping (EOCC) for each coupler
    # M         = vector of mass of cars
    # x2all     = vector of coupler displacements (common for all couplers) - (nc x 1)
    # Fmax2all  = hysterisis maximum coupler force associated with displacement for each coupler - (nc x ncouplers)
    # Fmin2all  = hysterisis minimum coupler force associated with displacement for each coupler - (nc x ncouplers)
    # Klocked   = vector of locked (between min & max hysterisis limits) coupler spring constant
    # Kawu      = anti-windup gain (hysterisis modelled as saturation with anti-windup)
    #
    # Xdot      = current state derivatives
    # Frope     = current rope forces as used (interpolated)
    # F         = coupler coupler forces (nx x 1) 
    # Fmin      = min force of hysteresis at current displacement
    # Fmax      = max force of hysteresis at current displacement
    #
    # H. Kirk Mathews (GE Research) 18 July 2018
    # Joe Wakeman     (Wabtec)      31 July 2024 - Translated to Python
    
    Xdot = np.zeros(3*n)
    F = np.zeros(n)
    a = np.zeros(n)
    e = np.zeros(n)
    
    # look up rope force at current time, t,  **ASSUME CONSTANT TIME STEP**
    dt = T[1] - T[0]
    i = min(len(T), max(0,int(round((t-T[0])/dt))))
    Frope = FROPE[i,:]

    # number of points in force-displacement lookup table
    # assumes table has been interpolated to a constant step size for quick
    # lookup
    m = len(x2all)
    dx = x2all[1]-x2all[0]

    Fmin = np.zeros(len(Frope))
    Fmax = np.zeros(len(Frope))

    for i in range(n): # each coupler
        # extract the current states from the state vector, X
        DX = X[i+2*n]
        DV = X[i+n]
        F1 = X[i]

        # find location in force-displacement table that brackets the current
        # displacement. DX is x2all(idx0) and x2all(idx1)
        # ASSUMES CONSTANT DISPLACEMENT STEP SIZE 
        idx = (DX-x2all[0])/dx
        idx0 = int(math.floor(idx))  # Ensure integer type like MATLAB
        frac = idx - idx0  # fractional location of DX between x2all(idx0) and x2all(idx1)
        idx1 = idx0 + 1
        
        # Bounds checking to match MATLAB logic
        if idx0 < 0:
            idx0 = 0
            idx1 = 1
        elif idx0 > m-2:
            idx0 = m-2
            idx1 = m-1

        # linear interpolation of force-displacement table
        # min/max force at current displacement, DX
        # Use explicit type conversion to match MATLAB precision
        Fmin[i] = float(Fmin2all[idx0,i]) * (1.0-frac) + float(Fmin2all[idx1,i]) * frac
        Fmax[i] = float(Fmax2all[idx0,i]) * (1.0-frac) + float(Fmax2all[idx1,i]) * frac

        # Limit the coupler force state, Fmin <= F1 <= Fmax
        if F1 < Fmin[i]:
            F[i] = Fmin[i]
        elif F1 > Fmax[i]:
            F[i] = Fmax[i]
        else:
            F[i] = F1

        # sum of forces on the i'th vehicle
        # Ensure consistent floating point operations to match MATLAB
        e[i] = float(F[i]) - float(Frope[i]) + float(b[i])*float(DV) + float(c[i])*float(DV)*float(DX)*float(F[i])
        # print(f'e[{i}]: {e[i]:.17f}')

    # compute the absolute acceleration of each car
    a[0] = e[0]/M[0]
    for i in range(1,n): # each coupler
        a[i] = (e[i]-e[i-1])/M[i]

    # form the state derivatives
    # F1dot = locked stiffness*DV + anti-windup to keep F1 close to F, when F is limited
    # DVdot = DA = accel(car i+1)-accel(car i)
    # DXdot = DV
    for i in range(n-1):
        Xdot[i]     = Klocked[i]*X[i+n] + Kawu[i]*( F[i] - X[i] )  # F1dot
        Xdot[i+n]   = a[i+1]-a[i];                                 # DVdot
        Xdot[i+2*n] = X[i+n];                                      # DXdot

    i = n-1
    Xdot[i]   = Klocked[i]*X[2*i] + Kawu[i]*( F[i] - X[i] )
    Xdot[2*i] = 0
    Xdot[3*i] = X[2*i]

    #np.savetxt("Xdot_out.csv", Xdot, delimiter=",")
    #np.savetxt("Frope_out.csv", Frope[np.newaxis], delimiter=",", fmt='%f')
    #np.savetxt("F_out.csv", F[np.newaxis], delimiter=",", fmt='%f')
    #np.savetxt("Fmin_out.csv", Fmin[np.newaxis], delimiter=",", fmt='%f')
    #np.savetxt("Fmax_out.csv", Fmax[np.newaxis], delimiter=",", fmt='%f')

    return Xdot, Frope, F, Fmin, Fmax