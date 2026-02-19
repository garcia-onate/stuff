import numpy as np
from scipy.interpolate import interp1d

def upsampleProfile(profile, dist):
    dist = dist.flatten()
    profileout = {}
    
    names = list(profile.keys())
    for i in range(len(names)):
        
        if names[i] != 'Dist':
            
            if isinstance(profile[names[i]], dict):
                # structure - don't interpolate
                profileout[names[i]] = profile[names[i]]
                continue
            
            profile_field = np.array(profile[names[i]])
            n, m = profile_field.shape if profile_field.ndim > 1 else (len(profile_field), 1)
            
            if m == len(profile['Dist']):
                # transpose
                profile[names[i]] = profile_field.T
                profile_field = profile[names[i]]
                n, m = profile_field.shape if profile_field.ndim > 1 else (len(profile_field), 1)
            
            if n == len(profile['Dist']):
                # nx1, same size as dist therefore interpolate
                if names[i] in ['fence_flg', 'FencePos', 'PowerPlanState']:
                    # integer - use previous value interpolation
                    f = interp1d(profile['Dist'], profile_field.astype(float), 
                               kind='previous', fill_value='extrapolate', axis=0)
                    profileout[names[i]] = f(dist)
                else:
                    # linear interpolation
                    f = interp1d(profile['Dist'], profile_field.astype(float), 
                               kind='linear', fill_value='extrapolate', axis=0)
                    profileout[names[i]] = f(dist)
            else:
                # different size - don't interpolate
                profileout[names[i]] = profile[names[i]]
    
    profileout['Dist'] = dist
    # profileout['CouplerForcesKips'] = calcForcesFromProfile(spec,profileout);
    
    return profileout