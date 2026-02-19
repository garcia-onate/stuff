# To gain speed FastSim lumps cars into groups. This function interpolates
# the resulting coupler forces (at the boundaries of the groups) to each
# coupler within each group. It uses a modification of the rope model that
# includes non-zero boundary conditions.
#
# Usage: fs = interpFastSim(fs,spec)
#
# fs    FastSim output structure (new fields added of the form xx_allcars)
# spec  standard TO spec structure defining the train and trip
#
# H. Kirk Mathews (GE Research) 12 Oct 2018

import numpy as np
from scipy.interpolate import interp1d
from .expandSpec import expandSpec

def interpFastSim(fs, spec, TrackData=None, Locos=None):
    
    # extract the needed fields of the FastSim results (for code clarity)
    F = fs['Fsi']          # coupler forces of groups
    F1 = fs['F1si']        # coupler forces of groups - pre-limiting
    Frope_in = fs['Frope_insi']   # rope force input to FastSim
    cargroup = fs['cargroup']     # car groups
    
    # interpolate input rope forces (for each car) to simulation output time
    Frope_in1 = interp1d(fs['t_in'], Frope_in, axis=0, kind='linear', 
                         fill_value='extrapolate')(fs['t'])
    
    xSpec = expandSpec(spec, TrackData, Locos)  # expand to spec to extract route and train information from the database
    
    # Interpolate the forces within groups using rigid coupler calculation
    # TODO - make this a reusable function
    cumWeight = np.cumsum(xSpec['Train']['weightsAll'])
    
    # Create interpolation points: [0, cumWeight[cargroup]]
    interp_points = np.concatenate([[0], cumWeight[np.array(cargroup) - 1]])
    
    # Create force values for interpolation: [F[:,0]*0, F-Frope_in1[:,cargroup]]
    F_zero = F[:, 0:1] * 0
    F_interp_vals = np.concatenate([F_zero, F - Frope_in1[:, np.array(cargroup) - 1]], axis=1)
    F1_interp_vals = np.concatenate([F1[:, 0:1] * 0, F1 - Frope_in1[:, np.array(cargroup) - 1]], axis=1)
    
    # Interpolate forces for all cars
    fs['Fsi_allcars'] = np.zeros((F.shape[0], len(cumWeight)))
    fs['F1si_allcars'] = np.zeros((F1.shape[0], len(cumWeight)))
    
    for i in range(F.shape[0]):
        fs['Fsi_allcars'][i, :] = interp1d(interp_points, F_interp_vals[i, :], 
                                          kind='linear', fill_value='extrapolate')(cumWeight)
        fs['F1si_allcars'][i, :] = interp1d(interp_points, F1_interp_vals[i, :], 
                                           kind='linear', fill_value='extrapolate')(cumWeight)
    
    # Add rope forces back
    fs['Fsi_allcars'] = fs['Fsi_allcars'] + Frope_in1
    fs['F1si_allcars'] = fs['F1si_allcars'] + Frope_in1
    fs['Fropesi_allcars'] = Frope_in1
    
    ncarsingroup = np.diff(np.concatenate([[0], cargroup]))
    
    # Interpolate displacements
    DXsi_normalized = fs['DXsi'].T / ncarsingroup[:, np.newaxis]
    fs['DXsi_allcars'] = interp1d(cargroup, DXsi_normalized, axis=0, kind='next', 
                                 fill_value='extrapolate')(np.arange(1, len(cumWeight) + 1)).T
    
    # Initialize velocities
    fs['DVsi_allcars'] = fs['DXsi_allcars'] * 0
    fs['DVsi_allcars'][:, np.array(cargroup) - 1] = fs['DVsi']
    
    if 'fminsi' in fs:
        fs['fminsi_allcars'] = interp1d(cargroup, fs['fminsi'].T, axis=0, kind='next', 
                                       fill_value='extrapolate')(np.arange(1, len(cumWeight) + 1)).T
    
    if 'fmaxsi' in fs:
        fs['fmaxsi_allcars'] = interp1d(cargroup, fs['fmaxsi'].T, axis=0, kind='next', 
                                       fill_value='extrapolate')(np.arange(1, len(cumWeight) + 1)).T
    
    return fs