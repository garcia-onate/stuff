# lookup reference train and track info and add to the spec structure

import numpy as np
from .makeeffelev import makeeffelev

def expandSpec(xSpec, TrackData, Locos,):
    
    xSpec['TrackData'] = TrackData
    xSpec['locoParam'] = Locos
    
    xSpec['Train']['numLocos'] = len(xSpec['Consist'])
    xSpec['Train']['numAll'] = xSpec['Train']['numLocos'] + xSpec['Train']['NumCars']
    
    xSpec['Train']['Weight'] = sum([loco['Weight'] for loco in xSpec['locoParam']]) + sum(xSpec['Train']['LoadWeight'])
    
    xSpec['TrackData']['EffElev'] = makeeffelev(xSpec['TrackData']['Grade'], xSpec['TrackData']['PathDist'])
    
    isCar = np.ones(xSpec['Train']['numAll'], dtype=bool)
    isLoco = np.zeros(xSpec['Train']['numAll'], dtype=bool)
    isLoco[np.array(xSpec['Train']['LocoPosition']) - 1] = True  # Convert to 0-based indexing
    isCar[np.array(xSpec['Train']['LocoPosition']) - 1] = False
    xSpec['Train']['isCar'] = isCar
    xSpec['Train']['isLoco'] = isLoco
    
    noPreLoad = 400 * 1000
    
    CouplerTypeAll = np.zeros(xSpec['Train']['numAll'])
    CouplerTypeAll[isCar] = xSpec['Train']['CouplerType']
    CouplerTypeAll[isLoco] = 0
    xSpec['Train']['CouplerTypeAll'] = CouplerTypeAll
    
    preLoadAll = np.zeros(xSpec['Train']['numAll'])
    preLoadAll[isCar] = xSpec['Train']['PreLoad']
    preLoadAll[isLoco] = noPreLoad
    xSpec['Train']['preLoadAll'] = preLoadAll
    
    weightsAll = np.zeros(xSpec['Train']['numAll'])
    weightsAll[isLoco] = [loco['Weight'] for loco in xSpec['locoParam']]
    weightsAll[isCar] = xSpec['Train']['LoadWeight']
    xSpec['Train']['weightsAll'] = weightsAll
    
    lengthsAll = np.zeros(xSpec['Train']['numAll'])
    lengthsAll[isLoco] = [loco['Length'] for loco in xSpec['locoParam']]
    lengthsAll[isCar] = xSpec['Train']['LoadLength']
    xSpec['Train']['lengthsAll'] = lengthsAll
    
    return xSpec