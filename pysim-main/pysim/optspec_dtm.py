# Apply DTM logic to OptSpec.TrainParam.LoadWeight and
# OptSpec.TrainParam.LoadLength to get OptSpec.TrainParam.TareWeight
# and OptSpec.TrainParam.GrossWeight.  Also deals with the case where
# the input is a spec

import numpy as np

def optspec_dtm(OptSpec):
    
    tp = OptSpec['Train']
    
    tp['GrossWeight'] = tp['LoadWeight'].copy()
    tp['TareWeight'] = tp['LoadWeight'].copy()
    for k in range(len(tp['LoadWeight'])):
        w = tp['LoadWeight'][k]
        len_val = tp['LoadLength'][k]
        if len_val < 100:
            gw = max(w, 140)
        else:
            gw = max([1.64 * len_val - 104, 140, w])
        tw_est = gw / 4.2
        if tw_est > w:
            tw = w
            gw = tw * 4.2
        else:
            tw = tw_est
        tp['GrossWeight'][k] = gw
        tp['TareWeight'][k] = tw

    OptSpec['Train'] = tp
    
    return OptSpec