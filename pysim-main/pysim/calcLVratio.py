# Function to calculate an approximate L/V force ratio ignoring
# superelevation and centripital acceleration given coupler forces.
#
# Usage: LVratio = calcLVratio(spec,Flb,Dist)
#
# spec  TO MATLAB spec structure
# Flb   Coupler forces in pounds vs distance
# Dist  Distance along the track from start of the trip
#
# H. Kirk Mathews (GE Research) 15 Oct 2018

import numpy as np
from scipy.interpolate import interp1d
from .expandSpec import expandSpec

def calcLVratio(spec, TrackData, Locos, Flb, Dist):
    
    # expand the spec to get track and train information
    xSpec = expandSpec(spec, TrackData, Locos)
    
    Length = xSpec['Train']['lengthsAll']
    Weight = xSpec['Train']['weightsAll'] * 2000
    cumlen = np.concatenate([[0], np.cumsum(Length)]) / 5280  # mile
    distCurve = xSpec['TrackData']['Grade']['Dist']
    Curvature = xSpec['TrackData']['Grade']['Curvature']
    
    hundredfoot = 100
    LVratio = np.zeros(Flb.shape)
    
    for i in range(Flb.shape[1]):
        # car(i-1)--coupler(i-1)--car(i)--coupler(i)--car(i+1)--coupler(i+1)
        #
        # car & coupler i
        D1 = interp1d(distCurve, Curvature, kind='next', fill_value='extrapolate')(Dist - cumlen[i])  # degree of curvature
        R1 = 180 / np.pi * hundredfoot / D1  # radius of curvature
        F1 = Flb[:, i]    # coupler force
        L1 = Length[i]    # car length
        
        # car & coupler i-1
        D0 = np.zeros(Dist.shape)
        F0 = Flb[:, 0] * 0
        L0 = 0
        if i > 0:
            D0 = interp1d(distCurve, Curvature, kind='next', fill_value='extrapolate')(Dist - cumlen[i-1])  # degree of curvature
            F0 = Flb[:, i-1]    # coupler force
            L0 = Length[i-1]    # car length
        R0 = 180 / np.pi * hundredfoot / D0  # convert degree-of-curvature to radius
        
        # car & coupler i+1
        D2 = np.zeros(Dist.shape)
        L2 = 0
        if i < Flb.shape[1] - 1:
            D2 = interp1d(distCurve, Curvature, kind='next', fill_value='extrapolate')(Dist - cumlen[i+1])  # degree of curvature
            L2 = Length[i+1]    # car length
        R2 = 180 / np.pi * hundredfoot / D2  # convert degree-of-curvature to radius
        
        Flat1 = -0.5 * F1 * (L1 / R1 + L2 / R2)  # lateral force from coupler i
        Flat0 = -0.5 * F0 * (L0 / R0 + L1 / R1)  # lateral force from coupler i-1
        Flat = Flat1 + Flat0                     # total laterial force
        LVratio[:, i] = -Flat / Weight[i]        # L/V force ratio
    
    return LVratio