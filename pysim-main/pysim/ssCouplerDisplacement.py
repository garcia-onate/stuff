import numpy as np

def ssCouplerDisplacement(x2all, Fmin2all, Fmax2all, Frope):
    # Calculate the steady-state coupler displacement from the first value of 
    # the rope force, choosing Fmax if the rope force is increasing and Fmin if
    # it is decreasing.
    #
    # Usage: x = ssCouplerDisplacement(x2all,Fmin2all,Fmax2all,Frope)
    #
    # x2all     vector of coupler displacements (common for all couplers) - (ntable x 1)
    # Fmax2all  hysterisis maximum coupler force associated with displacement for each coupler - (ntable x ncouplers)
    # Fmin2all  hysterisis minimum coupler force associated with displacement for each coupler - (ntable x ncouplers)
    # Klocked   vector of locked (between min & max hysterisis limits) coupler spring constant
    # Frope     rope forces for couplers vs time or distance
    # x         the steady-state displacement the results in the input rope force, Frope   
    #
    # H. Kirk Mathews (GE Research) 14 Nov 2018

    x = np.zeros(Fmax2all.shape[1])
    
    for i in range(Fmax2all.shape[1]):
        d = np.sign(np.diff(Frope[0:2, i]))                    # the derivative of the rope force determins which part of the hystersis we're on
        e = np.max(np.abs(Fmax2all[:, i])) * np.finfo(float).eps * 2  # add a small insignificant slope to the table to force a unique solution
                                                               # (prevents interp1 from failing if the Fmax or Fmin has zero slope, i.e., non-unique values)
        if d < 0:
            # Frope decreasing therefore we're on the MINIMUM side of the hystersis envelope
            x[i] = np.interp(Frope[0, i], Fmin2all[:, i] + np.arange(1, Fmin2all.shape[0] + 1) * e, x2all)
        else:
            # Frope decreasing therefore we're on the MAXIMUM side of the hystersis envelope
            x[i] = np.interp(Frope[0, i], Fmax2all[:, i] + np.arange(1, Fmax2all.shape[0] + 1) * e, x2all)
    
    return x