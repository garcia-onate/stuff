# USAGE: F = locoeffort_of_notch_speed(Loco, notch, v);
# PURPOSE: Calculate the effort from notch and speed using a simplified
#          locomotive model, where braking effort vs speed has the same
#          shape as motoring effort vs speed.
# INPUTS: Loco
#              Structure containing at least: 
#                1. THP   (which should be Tractive horsepower times
#                traction efficiency, a vector of length 17
#                2. MaxTE (Maximum TE in pounds)
#                3. Notch (Notch vector, normally -8:1:8)

import numpy as np
from scipy.interpolate import interp1d

def locoeffort_of_notch_speed(Loco, notch, v):
    
    # small speed, to prevent division by zero
    small_spd = 1e-3
    
    notch = notch.flatten()
    v = v.flatten()
    # Prevent v from being small
    v = np.maximum(v, small_spd)
    
    # Power
    p = interp1d(Loco['Notch'], Loco['THP'], kind='linear', fill_value='extrapolate')(notch)
    # max effort
    Fmax = np.abs(interp1d(Loco['Notch'], Loco['MaxTE'], kind='linear', fill_value='extrapolate')(notch))
    # Initial estimate of effort
    F1 = 375 * p / v
    # Sign of the force vector
    sgn_f = np.sign(F1)
    # Calculate effort
    F = np.minimum(np.abs(F1), Fmax) * sgn_f
    
    return F