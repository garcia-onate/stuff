# Reduces the size of the FastSim output by down sampling the forces to a 
# specified sample time.  It uses a sign-aware moving max function to 
# preserve the worst-case magnitude of the forces when down sampling.
#
# Usage: fs = downSampleFastSim(fs,Ts)
#
# fs = input FastSim structure
# Ts = desired sample time.
#
# fs = input FastSim structure
#
# H. Kirk Mathews (GE Research) 30 June 2020

import numpy as np
from scipy.ndimage import maximum_filter1d

def downSampleFastSim(fs, Ts):
    
    n = fs['Fsi'].shape[0]       # number of time and distance points
    dt = fs['t'][1] - fs['t'][0]  # original sample time
    
    m = max(1, round(Ts / dt))   # downsample ratio
    
    # downsample each field of length "n" in the FastSim structure, fs
    fn = list(fs.keys())
    for i in range(len(fn)):
        # Only process numpy arrays that have the right length
        if isinstance(fs[fn[i]], np.ndarray) and fs[fn[i]].shape[0] == n:
            tmp = fs[fn[i]]  # default value to decimate
            if fn[i] in ['Fsi', 'F1si', 'Fropesi', 'Fsi_allcars', 'F1si_allcars', 'Fropesi_allcars', 'LVratio']:
                # moving max of forces so that decimation captures the worst case value of forces between down samples
                tmp = movmax(np.abs(tmp), [int(np.floor(m/2)), m-1-int(np.floor(m/2))], axis=0) * np.sign(tmp)
            
            # Handle both 1D and 2D arrays
            if tmp.ndim == 1:
                fs[fn[i]] = tmp[::m]  # decimate 1D array
            else:
                fs[fn[i]] = tmp[::m, :]  # decimate 2D array
    
    return fs


def movmax(data, window, axis=0):
    """
    Moving maximum filter equivalent to MATLAB's movmax function
    
    Parameters:
    data: input array
    window: [pre, post] window size or scalar for symmetric window
    axis: axis along which to apply the filter
    """
    if isinstance(window, list):
        pre, post = window
        window_size = pre + post + 1
        # Create padded array
        if axis == 0:
            padded = np.pad(data, ((pre, post), (0, 0)), mode='edge')
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                result[i] = np.max(padded[i:i+window_size], axis=0)
        else:
            padded = np.pad(data, ((0, 0), (pre, post)), mode='edge')
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                result[:, i] = np.max(padded[:, i:i+window_size], axis=1)
    else:
        # Symmetric window
        window_size = window
        result = maximum_filter1d(data, size=window_size, axis=axis, mode='constant')
    
    return result