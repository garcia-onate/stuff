# Purpose: Function to output train forces using rope model at every
# car (and loco) for every location in a plan
# Inputs: spec    Structure
#                spec must have the fields Consist, Route, Trip and
#                Train.
#                spec.Train must contain the fields LoadWeight,
#                Davis_a, Davis_b, Length, NumCars, and
#                LocoPosition. LoadWeight must 
#                either be a vector of length equal to the number
#                of cars (individual car weights) or a scalar
#                (total car weight). Optionally, there can be a
#                member called LoadLength (individual car
#                lengths). In this case, the length of the train is
#                computed from sum(LoadLength) + length of
#                locos. Any loco in the database without a length
#                is assumed to be 78 feet long.
#        profile Notch, Speed, Dist. Notch can have either one column
#        (for synchronous mode of operation) or as many cols as number of
#        locos (for async)
# Outputs: F      Matrix with ncars + nlocos cols and
#                length(profile.Dist) rows
#                (F(:,ncars+nlocos) corresponds to coupler n+1, which
#                doesn't exist and its force should always be zero.
# EDIT: GABRIEL: Adapting code to accept fence position > 1 and moving
# fence (Jun-2014)

import numpy as np
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from .gravityforces import gravityforces
from .slewRateFilter import slewRateFilter
from .GetHCBoundsDef_braking import GetHCBoundsDef_braking
from .GetESpec_braking import GetESpec_braking
from .locoeffort_of_notch_speed import locoeffort_of_notch_speed

def calcRopeForces(profile, TrackData, Train, Locos, slew=True):
    
    n = len(profile['Dist'])
    
    dataset = TrackData
    trainParam = Train
    locoParam = Locos
    numcars = trainParam['NumCars']
    numlocos = len(locoParam)
    locoposition = trainParam['LocoPosition']
    
    # calculate notch for each locomotive
    Notch = np.zeros((n, numlocos))
    iconsist = 1
    consist = np.zeros(numlocos, dtype=int)
    consist[0] = iconsist
    Notch[:, 0] = np.array(profile['leadNotch'])
    for iloco in range(1, numlocos):
        if locoposition[iloco] > locoposition[iloco-1] + 1:
            # new consist
            iconsist = iconsist + 1
        consist[iloco] = iconsist
        beforeFence = iconsist <= np.array(profile['fence_flg'])
        afterFence = ~beforeFence
        Notch[:, iloco] = np.array(profile['leadNotch']) * beforeFence + np.array(profile['remoteNotch']) * afterFence
    
    # Adding slew rate filtering
    nNotches = Notch.shape[1]
    if slew:
        for iNotch in range(nNotches):
            Notch[:, iNotch] = slewRateFilter(Notch[:, iNotch], np.array(profile['Time']), locoParam)
    
    Notch = Notch.reshape(n, numlocos)
    
    trainParam['Weight'] = sum([loco['Weight'] for loco in locoParam]) + sum(trainParam['LoadWeight'])
    
    # Compute efforts from different locos
    Speed = np.array(profile['Speed']).flatten()
    Dist = np.array(profile['Dist']).flatten()
    Time = np.array(profile['Time']).flatten()
    
    Flocos_only = np.zeros((n, numlocos))
    z = GetHCBoundsDef_braking()
    Fuel = np.zeros(Notch[:, 0].shape)
    for k in range(numlocos):
        # Warning: Use simple DB models by default
        espec = GetESpec_braking(trainParam, [locoParam[k]], z['Def'])
        Flocos_only[:, k] = locoeffort_of_notch_speed(espec['Loco'], Notch[:, k], Speed)
        FuelRate = interp1d(espec['Loco']['Notch'], espec['Loco']['FuelRate'], 
                           kind='linear', fill_value='extrapolate')(Notch[:, k])
        Fuel = Fuel + cumtrapz(Time, FuelRate, initial=0)
    
    Fgrav = gravityforces(dataset, trainParam, locoParam, Dist)
    Fgravloco = Fgrav.copy()
    Floco = Fgrav * 0
    Fgravloco[:, np.array(locoposition) - 1] = Fgravloco[:, np.array(locoposition) - 1] + Flocos_only
    Floco[:, np.array(locoposition) - 1] = Flocos_only
    
    # lococarweight
    locoweight = np.array([loco['Weight'] for loco in locoParam])
    carweight = trainParam['LoadWeight']
    lococarweight = np.zeros(numlocos + numcars)
    numlococars = len(lococarweight)
    carct = 0
    lococt = 0
    for k in range(numlocos + numcars):
        if (lococt < numlocos) and (locoposition[lococt] == k + 1):
            lococarweight[k] = locoweight[lococt]
            lococt = lococt + 1
        else:
            lococarweight[k] = carweight[carct]
            carct = carct + 1
    
    # Total drag forces
    Fdrag_tot = trainParam['Davis_a'] + trainParam['Davis_b'] * Speed + trainParam['Davis_c'] * Speed * Speed
    Fdrag_tot = -Fdrag_tot.flatten()
    
    # Drag is in pounds per ton
    Fdrag = np.tile(Fdrag_tot[:, np.newaxis], (1, numlococars)) * np.tile(lococarweight[np.newaxis, :], (n, 1))
    
    # Inertial forces
    # Acceleration, in g's
    # body forces
    Fbody = np.sum(Fgravloco + Fdrag, axis=1)
    a = Fbody / trainParam['Weight'] / 2000
    Finertia = -2000 * np.tile(lococarweight[np.newaxis, :], (n, 1)) * np.tile(a[:, np.newaxis], (1, numlococars))
    
    F = Fgravloco + Fdrag + Finertia
    
    # Finally, cumsum
    F = np.cumsum(F, axis=1)
    
    # Outputting rope forces due to grav, drag and loco
    FtotalLoco = np.sum(Floco, axis=1)
    FtotalGrav = np.sum(Fgrav, axis=1)
    FtotalDrag = np.sum(Fdrag, axis=1)
    aLoco = FtotalLoco / trainParam['Weight'] / 2000
    aGrav = FtotalGrav / trainParam['Weight'] / 2000
    aDrag = FtotalDrag / trainParam['Weight'] / 2000
    FinertiaLoco = -2000 * np.tile(lococarweight[np.newaxis, :], (n, 1)) * np.tile(aLoco[:, np.newaxis], (1, numlococars))
    FinertiaGrav = -2000 * np.tile(lococarweight[np.newaxis, :], (n, 1)) * np.tile(aGrav[:, np.newaxis], (1, numlococars))
    FinertiaDrag = -2000 * np.tile(lococarweight[np.newaxis, :], (n, 1)) * np.tile(aDrag[:, np.newaxis], (1, numlococars))
    Floco = Floco + FinertiaLoco
    Fgrav = Fgrav + FinertiaGrav
    Fdrag = Fdrag + FinertiaDrag
    
    Floco = np.cumsum(Floco, axis=1)
    Fgrav = np.cumsum(Fgrav, axis=1)
    Fdrag = np.cumsum(Fdrag, axis=1)
    
    return F, Fuel, Flocos_only, Fgravloco, Fgrav, Fdrag, Finertia, Floco