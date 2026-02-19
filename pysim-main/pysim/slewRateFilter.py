import numpy as np

def slewRateFilter(notch, time, locoParam=None):
    
    # Adapting Code to accept more than one column of notches 
    
    if locoParam is None:
        slewRate = 1000 * np.ones(17)
    else:
        ghp = np.flip(np.array(locoParam[0]['GHP'][:9]))
        hpsr = np.array(locoParam[0]['HPpersec'])
        deltaghp = np.diff(ghp)
        notchsr = 3600 * hpsr / deltaghp
        slewRate = np.concatenate([1000 * np.ones(9), np.flip(notchsr)])
    
    # slewRate = 1000; % notches per hour
    if notch.ndim == 1:
        # Handle 1D array (single column)
        numPoints = len(notch)
        for iPoint in range(1, numPoints):
            change = np.abs(notch[iPoint] - notch[iPoint-1])
            signal = np.sign(notch[iPoint] - notch[iPoint-1])
            timedelta = time[iPoint] - time[iPoint-1]
            iNotch = int(np.round(notch[iPoint-1])) + 8
            
            if change / timedelta > slewRate[iNotch]:
                notch[iPoint] = notch[iPoint-1] + signal * slewRate[iNotch] * timedelta
    else:
        # Handle 2D array (multiple columns)
        numPoints, numCols = notch.shape
        
        for iPoint in range(1, numPoints):
            change = np.abs(notch[iPoint, :] - notch[iPoint-1, :])
            signal = np.sign(notch[iPoint, :] - notch[iPoint-1, :])
            timedelta = time[iPoint] - time[iPoint-1]
            iNotch = np.round(notch[iPoint-1, :]).astype(int) + 8
            
            isSlewRate = change / timedelta > slewRate[iNotch]
            
            for iCol in range(numCols):
                if isSlewRate[iCol]:
                    notch[iPoint, iCol] = notch[iPoint-1, iCol] + signal[iCol] * slewRate[iNotch[iCol]] * timedelta
    
    return notch