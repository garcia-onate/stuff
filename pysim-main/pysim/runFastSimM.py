# Simulate in-train forces given a profile,  spec, and some parameters and 
# returns the forces along with other information in a structure.
# It calculates the rope forces from the profile and spec (the primary
# input to FastSim), initializes and runs FastSim.
#
# Usage: out = runFastSimM(profile, spec, dcar, Ts, dx, range, X0)
#
# profile = standard MATLAB TO planner profle
# spec    = standard MATLAB TO spec structure
# dcar    = nominal number of cars grouped to make FastSim fast
# Ts      = sample time to provide outputs
# dx      = distance (miles) step size to interpolate profile (default: 0.1 mile)
# range   = distance range to simulate = [startDist,endDist]
#           (default: first and last distance in profile)
# X0      = Initial states ([F;DV;DX] or options, 'zero', 'steady-state')
#           scalar value is expended, default - 'steady-state'
# 
# out     = FastSim output structure in SI units
#           Fields: Fsi         = Coupler force vector
#                   F1si        = Coupler force state (force before limiting)
#                   DAsi        = Relative acceleration between cars
#                   DXsi        = Relative distance between cars
#                   DVsi        = Relative speed between cars
#                   Fropesi     = Rope force used
#                   t           = time
#                   dist        = distance
#                   Fminsi      = Fmin of force-displacement table lookup used
#                   Fmaxsi      = Fmax of force-displacement table lookup used
#                   xminsi      = displacement associated with Fminsi of force-displacement table lookup used
#                   xmaxsi      = displacement associated with Fminsi of force-displacement table lookup used
#                                 (legacy, identical to xmaxsi)
#                   profile     = input profile
#                   dtsim       = FastSim output sample time
#                   Frope_insi  = Rope passed into FastSim
#                   t_in        = time associated with Fropw_insi
#                   cargroup    = car groups

import numpy as np
import time
from scipy.interpolate import interp1d
from .unit_conversions import lbf_, hour_, sec_
from .normalizeProfile import normalizeProfile
from .calcRopeForces import calcRopeForces
from .initFastSim import initFastSim
from .FastSimM import FastSimM

def runFastSimM(profile, spec, TrackData, Train, Locos, dcar=None, Ts=None, dx=None, range_=None, X0=None):
    
    # default miles to interpolate profile
    if dx is None:
        dx = 0.1
    
    # define start/stop distance
    if range_ is None:
        distStart = profile['Dist'][0]
        distEnd = profile['Dist'][-1]
    else:
        distStart = range_[0]
        distEnd = range_[1]
    
    # default initialization
    if X0 is None:
        X0 = 'steady-state'
    
    # interpolate the profile to fixed step size
    if dx is not None:
        profile = normalizeProfile(profile, dx)
    
    ## Loading profile definitions
    Dist_in = profile['Dist']
    t_in = np.concatenate([profile['Time'][:-1], [profile['Time'][-2] * 1.001]]) * 3600
    
    # Evaluate rope forces
    F, Fuel, Flocos_only, Fgravloco, Fgrav, Fdrag, Finertia, Floco = calcRopeForces(profile, TrackData, Train, Locos)  # preferred way, includes moving fence and rate limits
    
    # preparing the inputs for the FastSim
    Frope_in0 = F * lbf_()
    n0 = Frope_in0.shape[1]
    
    i1 = np.where(Dist_in >= distStart)[0]
    if len(i1) > 0:
        i1 = i1[0]
    else:
        i1 = 0
    
    i2 = np.where(Dist_in > distEnd)[0]
    if len(i2) > 0:
        i2 = i2[0]
    else:
        i2 = len(Dist_in)
        distEnd = Dist_in[-1]
    
    tstart = t_in[i1]
    tend = t_in[i2]
    
    # get the parameters required by FastSim, m-file version
    Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB, cargroup, x2all, Fmin2all, Fmax2all = initFastSim(spec, Locos, dcar, Ts)
    
    Frope_in = Frope_in0
    
    # interpolate the rope forces to a constant time step to speed up
    # interpolation in FastSimODE
    tTs = np.arange(tstart, tend + Ts, Ts)
    Fropesi = interp1d(t_in, Frope_in[:, np.array(cargroup) - 1], axis=0, kind='linear', 
                      fill_value='extrapolate')(tTs)
    
    n = len(cargroup)        # number of couplers for grouped car model
    M = groupMassVector      # mass vector of grouped cars
    b = groupDampingB        # linear damping term of grouped cars
    c = groupCushioningDampingB  # non-linear damping term of grouped cars (for EOCCs)
    
    ## simulate and time
    tic = time.time()
    tode, Xode, stats, Yode = FastSimM(tTs, Fropesi, tstart, tend, n, b, c, M, x2all, Fmax2all, Fmin2all, Klocked, Kawu, X0)
    dtsim = time.time() - tic
    
    # extract results
    t = tode
    DX = Xode[:, (np.arange(n) + 2*n)]
    DV = Xode[:, (np.arange(n) + n)]
    F1 = Xode[:, :n]
    DA = np.full(DX.shape, np.nan)
    
    # enforce force-displacement limits (may not be required anymore)
    F = np.zeros_like(F1)
    for ii in range(n):
        fmax = interp1d(x2all, Fmax2all[:, ii], kind='linear', fill_value='extrapolate')(DX[:, ii])
        fmin = interp1d(x2all, Fmin2all[:, ii], kind='linear', fill_value='extrapolate')(DX[:, ii])
        F[:, ii] = np.maximum(fmin, np.minimum(fmax, F1[:, ii]))
    
    # interpolate distance to FastSim output time
    dist = interp1d(profile['Time'] * hour_(), profile['Dist'], kind='linear', 
                   fill_value='extrapolate')(t * sec_())
    
    # FastSim output structure (SI units)
    out = {}
    out['Fsi'] = F
    out['F1si'] = F1
    out['DAsi'] = DA
    out['DXsi'] = DX
    out['DVsi'] = DV
    out['Fropesi'] = Fropesi
    out['t'] = t
    out['dist'] = dist
    out['Fminsi'] = Fmin2all
    out['Fmaxsi'] = Fmax2all
    out['xminsi'] = x2all
    out['xmaxsi'] = x2all
    out['profile'] = profile
    out['dtsim'] = dtsim
    out['Frope_insi'] = Frope_in
    out['t_in'] = t_in
    out['cargroup'] = cargroup

    # Output Fields Explanation
    # out['Fsi'] = F - Coupler Forces (Newtons)

    # Limited coupler forces between car groups after applying force-displacement constraints
    # These are the actual forces transmitted through the couplers
    # Units: Newtons (N) - SI force units
    # out['F1si'] = F1 - Unconstrained Coupler Forces (Newtons)

    # Raw coupler force states before applying force-displacement limits
    # These represent what the forces would be without physical coupler limitations
    # Units: Newtons (N)
    # out['DAsi'] = DA - Relative Acceleration (m/s²)

    # Relative acceleration between adjacent car groups
    # Currently filled with NaN values in this implementation
    # Units: meters per second squared (m/s²)
    # out['DXsi'] = DX - Relative Displacement (meters)

    # Relative distance/displacement between adjacent car groups
    # Represents coupler extension/compression
    # Units: meters (m)
    # out['DVsi'] = DV - Relative Velocity (m/s)

    # Relative speed between adjacent car groups
    # Rate of change of coupler extension/compression
    # Units: meters per second (m/s)
    # out['Fropesi'] = Fropesi - Rope Forces Used (Newtons)

    # The rope forces that were interpolated and fed into FastSim
    # These are the driving forces for the simulation
    # Units: Newtons (N) - converted from lbf using lbf_() conversion factor
    # out['t'] = t - Time Vector (seconds)

    # Simulation time points
    # Units: seconds (s)
    # out['dist'] = dist - Distance Vector (miles)

    # Distance traveled corresponding to each time point
    # Interpolated from the input profile
    # Units: miles (the code interpolates from profile distance which appears to be in miles)
    # out['Fminsi'] = Fmin2all - Minimum Force Limits (Newtons)

    # Lower bounds of the force-displacement lookup table for each coupler
    # Used to constrain coupler forces
    # Units: Newtons (N)
    # out['Fmaxsi'] = Fmax2all - Maximum Force Limits (Newtons)

    # Upper bounds of the force-displacement lookup table for each coupler
    # Used to constrain coupler forces
    # Units: Newtons (N)
    # out['xminsi'] = x2all - Displacement Points (meters)

    # Displacement values associated with the force limits
    # Units: meters (m)
    # out['xmaxsi'] = x2all - Displacement Points (meters)

    # Same as xminsi (legacy field, identical values)
    # Units: meters (m)
    # out['profile'] = profile - Input Profile Structure

    # The normalized input profile data structure
    # Contains distance, time, speed, and other track profile information
    # out['dtsim'] = dtsim - Simulation Runtime (seconds)

    # Wall-clock time taken to run the FastSim simulation
    # Units: seconds (s)
    # out['Frope_insi'] = Frope_in - Input Rope Forces (Newtons)

    # Original rope forces before interpolation, converted to SI units
    # Units: Newtons (N)
    # out['t_in'] = t_in - Input Time Vector (seconds)

    # Time points associated with the input rope forces
    # Units: seconds (s) - converted from hours using hour_()
    # out['cargroup'] = cargroup - Car Group Indices

    # Array indicating which cars are grouped together for the simulation
    # Dimensionless indices
    
    return out