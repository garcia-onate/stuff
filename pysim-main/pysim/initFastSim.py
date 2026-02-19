# Function to initialize the m-file version of FastSim adapted from the
# original SIMULINK version.
#
# Usage: [Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB, cargroup, x2all, Fmin2all, Fmax2all] = initFastSim(spec, dcar, Ts)
#
# INPUTS
# spec                      standard TO spec structure defining the train and trip
# dcar                      nominal number of cars grouped together for computational efficiency
# Ts                        desired time step of the output        
#
# OUTPUTS 
# Klocked                   vector of locked (between min & max hysterisis limits) coupler spring constant
# Kawu                      anti-windup gain (hysterisis modelled as saturation with anti-windup)
# x2all                     vector of coupler displacements (common for all couplers) - (nc x 1)
# Fmin2all                  hysterisis minimum coupler force associated with displacement for each coupler - (nc x ncouplers)
# Fmax2all                  hysterisis maximum coupler force associated with displacement for each coupler - (nc x ncouplers)
# groupMassVector           mass of each group
# groupDampingB             linear damping of each group's coupler
# groupCushioningDampingB   non-linear damping of each group's coupler (used for EOCCs)
# cargroup                  ending position of each group
#
# H. Kirk Mathews (GE Research) 18 Jul 2018

import numpy as np
from scipy.interpolate import interp1d
from .optspec_dtm import optspec_dtm
from .defineCarGroups import defineCarGroups
from .defineCouplers import defineCouplers
from .unit_conversions import ton_, kips_, in_, sec_

def initFastSim(spec, Locos, dcar=None, Ts=None):
    
    # default number of cars in a group
    if dcar is None:
        dcar = 5
    
    # default time step of the output
    if Ts is None:
        Ts = 5
    
    ## Loading spec definitions
    spec = optspec_dtm(spec)  # Apply DTM logic
    numCars = spec['Train']['NumCars']
    numLocos = len(spec['Train']['LocoPosition'])
    locoPos = spec['Train']['LocoPosition']
    trainLength = spec['Train']['Length']
    loadWeight = spec['Train']['LoadWeight']
    consist = spec['Consist']
    
    trainParam = spec['Train']
    
    # creating parameters based on the spec
    locoParam = Locos
    locoLength = [loco['Length'] for loco in locoParam]
    
    # lococarweight and coupler types
    locoweight = [loco['Weight'] for loco in locoParam]
    carweight = loadWeight
    weightsAll = np.zeros(numLocos + numCars)
    
    # Retrieve coupler info
    couplerTypeCars = (np.array(trainParam['CouplerType']) == 0).astype(int) + (np.array(trainParam['CouplerType']) == 3).astype(int) * 2
    
    preLoadCars = np.array(trainParam['PreLoad'])
    preLoadCars[couplerTypeCars == 1] = 0  # ensure preload is zero for draft gears
    if 'Stroke' not in trainParam:
        # default EOCC stroke to 28 inches
        strokeCars = np.zeros(preLoadCars.shape)
        strokeCars[couplerTypeCars == 2] = 28
    else:
        strokeCars = np.array(trainParam['Stroke'])
    
    # Ensure correct couplerType assignment when full data is available
    isEOCCcars = couplerTypeCars > 1
    # isexpectedPreload = (preLoadCars[couplerTypeCars==2] == 100000) | (preLoadCars[couplerTypeCars==2] == 50000)
    # isexpectedStroke  = (strokeCars[couplerTypeCars==2] == 28)
    
    # if any(~isexpectedPreload):
    #     print('\nWARNING: EOCC preloads other than 50 or 100 kips is an experimental feature. Use with caution.\n', file=sys.stderr)
    # 
    # if any(~isexpectedStroke): 
    #     print('\nWARNING: EOCC strokes other than 28 inches is an experimental feature. Use with caution.\n', file=sys.stderr)
    
    if np.any(preLoadCars[isEOCCcars] == 50):
        print('WARNING: some EOCC coupler preloads = 50. Assuming in kips and changing to 50000).')
        preLoadCars[preLoadCars[isEOCCcars] == 100] = 50000
    
    if np.any(preLoadCars[isEOCCcars] == 100):
        print('WARNING: some EOCC coupler preloads = 100. Assuming in kips and changing to 100000.')
        preLoadCars[preLoadCars[isEOCCcars] == 100] = 100000
    
    if np.any(preLoadCars[isEOCCcars] < 1000):
        print('WARNING: some EOCC coupler preloads < 1 kip. Assuming preload is 100000.')
        preLoadCars[preLoadCars[isEOCCcars] < 1000] = 100000
    
    couplerTypeAll = np.zeros(numLocos + numCars)
    
    # build lococarweight and couplerTypeVector
    isLocoAll = np.zeros(numLocos + numCars, dtype=bool)
    isLocoAll[np.array(locoPos) - 1] = True  # Convert to 0-based indexing
    weightsAll[isLocoAll] = locoweight
    weightsAll[~isLocoAll] = carweight
    
    couplerTypeAll[isLocoAll] = 1  # locos are always a draft gear
    couplerTypeAll[~isLocoAll] = couplerTypeCars
    
    preLoadAll = np.zeros(numLocos + numCars)
    strokeAll = np.zeros(numLocos + numCars)
    preLoadAll[isLocoAll] = 0
    preLoadAll[~isLocoAll] = preLoadCars
    strokeAll[isLocoAll] = 0
    strokeAll[~isLocoAll] = strokeCars
    
    # find unique coupler types
    couplerParamAll = np.column_stack([preLoadAll, strokeAll])
    uCouplerParam, indexUniqueCouplerType = np.unique(couplerParamAll, axis=0, return_inverse=True)
    uPreLoad = uCouplerParam[:, 0]
    uStroke = uCouplerParam[:, 1]
    
    n0 = numCars + numLocos
    
    ## Calculating car grouping set
    # cargroup = dcar:dcar:n0;
    # cargroup = defineCarGroups(dcar,couplerTypeVector,'-verbose','-equalizeSize');
    cargroup = defineCarGroups(dcar, couplerTypeAll.astype(int), '-equalizeSize')
    
    if cargroup[-1] != n0:
        cargroup.append(n0)
    ngroups = len(cargroup)
    
    uniqueTypes = np.unique(indexUniqueCouplerType)
    groupMassVector = np.zeros(ngroups)
    numCouplerTypesInGroup = np.zeros((ngroups, len(uniqueTypes)))
    
    for igroup in range(ngroups):
        if igroup == 0:
            groupIndices = np.arange(0, cargroup[igroup])
        else:
            groupIndices = np.arange(cargroup[igroup-1], cargroup[igroup])
        groupMassVector[igroup] = np.sum(weightsAll[groupIndices]) * ton_()
        for ityp_idx, ityp in enumerate(uniqueTypes):
            numCouplerTypesInGroup[igroup, ityp_idx] = np.sum(indexUniqueCouplerType[groupIndices] == ityp)
    
    couplerTypePreloads_kips = uPreLoad / 1000  # convert to kips
    couplerTypeStrokes_in = uStroke  # keep in inches
    
    n = len(cargroup)
    
    # Expand preloads and strokes to match numCouplerTypesInGroup dimensions (ngroups x ntypes)
    ngroups, ntypes = numCouplerTypesInGroup.shape
    preloads_matrix = np.tile(couplerTypePreloads_kips.reshape(1, -1), (ngroups, 1))
    strokes_matrix = np.tile(couplerTypeStrokes_in.reshape(1, -1), (ngroups, 1))
    
    ## Computing coupler model parameters
    coupler = defineCouplers(numCouplerTypesInGroup, preloads_matrix, strokes_matrix)
    Klocked = coupler['grouped']['Klocked'] * kips_() / in_()
    F_common = coupler['grouped']['F'] * kips_()  # common force values (1D)
    xmax = coupler['grouped']['xmax'] * in_()  # positive quadrant only (ngroups x nforces)
    xmin = coupler['grouped']['xmin'] * in_()  # positive quadrant only (ngroups x nforces)
    groupDampingB = coupler['grouped']['B'] * kips_() / (in_() / sec_())
    groupCushioningDampingB = coupler['grouped']['Bcushion']
    
    Kawu = 1e-5 * Klocked / 5  # anti windup gain for coupler force limits
    
    # Create force matrices for each group (repeat F_common for each group)
    Fmax = np.tile(F_common.reshape(1, -1), (n, 1))  # ngroups x nforces
    Fmin = np.tile(F_common.reshape(1, -1), (n, 1))  # ngroups x nforces
    
    # form lookup tables for coupler force upper and lower limits
    # Create full hysteresis curves (negative and positive quadrants)
    # For upper envelope: negative forces from -Fmin, positive forces from +Fmax
    Fmax2 = np.concatenate([-np.fliplr(Fmin[:, 1:]), Fmax], axis=1).T
    xmax2 = np.concatenate([-np.fliplr(xmin[:, 1:]), xmax], axis=1).T
    # For lower envelope: negative forces from -Fmax, positive forces from +Fmin  
    Fmin2 = np.concatenate([-np.fliplr(Fmax[:, 1:]), Fmin], axis=1).T
    xmin2 = np.concatenate([-np.fliplr(xmax[:, 1:]), xmin], axis=1).T
    
    x2all = np.unique(np.concatenate([xmax2.flatten(), xmin2.flatten()]))  # merge displacement break points
    
    # interpolate min/max hystersis envelope to the merged displacements
    Fmax2all = np.zeros((len(x2all), n))
    Fmin2all = np.zeros((len(x2all), n))
    for i in range(n):
        Fmax2all[:, i] = interp1d(xmax2[:, i], Fmax2[:, i], kind='linear', fill_value='extrapolate')(x2all)
        Fmin2all[:, i] = interp1d(xmin2[:, i], Fmin2[:, i], kind='linear', fill_value='extrapolate')(x2all)
    
    return Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB, cargroup, x2all, Fmin2all, Fmax2all
