import numpy as np
from .upsampleProfile import upsampleProfile

def normalizeProfile(profile, step=0.25):
    
    # Fix range to avoid NaNs in the Y when interpolating
    range_vals = [0, 0]
    range_vals[0] = profile['Dist'][0]
    range_vals[1] = profile['Dist'][-1]
    
    dist_pattern = np.arange(range_vals[0], range_vals[1] + step, step)
    if dist_pattern[-1] < range_vals[1]:
        dist_pattern = np.append(dist_pattern, range_vals[1])
    
    profile_new = upsampleProfile(profile, dist_pattern)
    
    return profile_new