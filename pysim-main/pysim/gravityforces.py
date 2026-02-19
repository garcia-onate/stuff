import numpy as np
from .makeeffelev import makeeffelev

def gravityforces(dataset, trainParam, locoParam, distout):
    # Compute gravity forces on a train at the end of each car. Produce
    # a vector with as many cols as numcars + numlocos and as many rows
    # as the distout vector (computed at each point in distout)

    EffElev = makeeffelev(dataset['Grade'], dataset['PathDist'])
    dist = EffElev['Dist']
    effElev = EffElev['Values']

    # Initialize a few things
    locolength = np.array([loco['Length'] for loco in locoParam])
    N = len(locoParam) + len(trainParam['LoadWeight'])

    if np.max(trainParam['LocoPosition']) > N:
        raise ValueError('Loco position is beyond end of train')
    
    trainwt = np.zeros(N)
    trainwt[np.array(trainParam['LocoPosition']) - 1] = np.array([loco['Weight'] for loco in locoParam])
    non_loco_positions = np.setdiff1d(np.arange(N), np.array(trainParam['LocoPosition']) - 1)
    trainwt[non_loco_positions] = trainParam['LoadWeight']
    
    trainlen = np.zeros(N)
    trainlen[np.array(trainParam['LocoPosition']) - 1] = locolength
    trainlen[non_loco_positions] = trainParam['LoadLength']
    
    cumlen = np.concatenate([[0], np.cumsum(trainlen)]) / 5280  # mi
    F = np.zeros((len(distout), N))
    
    for i in range(N):
        F[:, i] = trainwt[i] * (np.interp(distout - cumlen[i], dist, effElev) - 
                               np.interp(distout - cumlen[i + 1], dist, effElev)) / trainlen[i]
    
    F = -F * 2000  # Pounds, change sign
    
    return F