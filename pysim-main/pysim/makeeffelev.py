import numpy as np

def makeeffelev(Grade, endDist=None):
    # effElev = makeeffelev(Grade[,endDist])
    # Generate an effective elevation structure from a grade data structure. If
    # optional endDist argument is given, output is specifed to distance of at
    # least endDist.
    #
    # by David S. K. Chan   6/20/2008
    #
    # Modified by Bryan Hermsen 10/4/2011:
    # created if/else structure for EffElev.Values. Track Databases I am using
    # all have Grade.Percent as column data. Created the if/else in case there
    # are some databases with row data. Not sure why this script was orginally
    # designed to assume row data...

    dist = np.array(Grade['Dist']).copy()
    if endDist is not None and dist[-1] < endDist:
        dist = np.append(dist, endDist)
    
    idx = np.arange(len(dist) - 1)
    rundist = (dist[idx + 1] - dist[idx]) * 5280
    
    EffElev = {}
    EffElev['Dist'] = dist
    
    # Handle both 1D and 2D arrays for Grade['Percent']
    percent = np.array(Grade['Percent'])
    curvature = np.array(Grade['Curvature']) if 'Curvature' in Grade else np.zeros_like(percent)
    
    # Ensure 1D arrays
    if percent.ndim > 1:
        if percent.shape[0] < percent.shape[1]:
            # Row vector - transpose to column
            percent = percent.flatten()
        else:
            # Column vector - flatten
            percent = percent.flatten()
    
    if curvature.ndim > 1:
        curvature = curvature.flatten()
    
    # Calculate effective elevation
    EffElev['Values'] = np.concatenate([[0], np.cumsum((percent[idx] + 0.04 * np.abs(curvature[idx])) / 100.0 * rundist)])
    
    return EffElev