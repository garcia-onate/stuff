# Defines the force-displacement hysteretic profile for each coupler 
# and creates a single merged force-displacement curve for grouped cars.
#
# Usage: coupler = defineCouplers(preloads,strokes)
#
# numTypesInGroup = number of each coupler type in each group (ngroups x ntypes) - type is defined by preload and stroke
# preloads        = preload of each coupler in the group (kips), zero means draft gear
# strokes         = strokes of each coupler in the group (kips), ignored for draft gear
#
# coupler = output structure with fields:
#           spec    = structure define each coupler (primarily: force-displacment min/max, Klocked)
#           common  = interpolated spec to a common displacement
#           inverse = inverse coupler lookup table, i.e., displacement-force, xmax/xmin for common force (used to merged grouped cars)
#           grouped = the coupler definition for each group
#
# H. Kirk Mathews (GE Research) 18 July 2018
#
# modified to specify any preload and stroke 4 Jan 2024 HKM

import numpy as np
from scipy.interpolate import interp1d

def defineCouplers(numTypesInGroup, preloads=None, strokes=None):
    
    preloadGiven = preloads is not None and len(preloads) > 0
    strokeGiven = strokes is not None and len(strokes) > 0
    
    if preloadGiven and not strokeGiven:
        strokes = np.full(np.array(preloads).shape, 28)
    
    elif not preloadGiven and strokeGiven:
        raise ValueError('preload must be given if stroke if given')
    
    elif not preloadGiven and not strokeGiven:
        # for backward compatibility
        numTypesInGroup = np.array(numTypesInGroup)
        ngroups, ntypes = numTypesInGroup.shape
        if ntypes < 3:
            numTypesInGroup = np.column_stack([numTypesInGroup, np.zeros((ngroups, 3 - ntypes))])
        ngroups, ntypes = numTypesInGroup.shape
        if ntypes <= 3:
            defPreload = [0, 100, 50]
            defStroke = [0, 28, 28]
            preloads = np.tile(defPreload[:ntypes], (ngroups, 1))
            strokes = np.tile(defStroke[:ntypes], (ngroups, 1))
        else:
            raise ValueError('when only numTypesInGroup is given it must be ngroups x ntypes, where ntypes = 1,2 or 3')
    
    preloads = np.array(preloads)
    strokes = np.array(strokes)
    numTypesInGroup = np.array(numTypesInGroup)
    
    assert preloads.shape == strokes.shape, 'preload and strokes must be the same size'
    assert preloads.shape[1] == numTypesInGroup.shape[1], 'preload and numTypesInGroup must have the same number of columns'
    
    spec = []
    
    for i in range(preloads.shape[1]):
        
        preload = preloads[0, i]
        stroke = strokes[0, i]
        
        spec_i = {}
        
        if preload == 0:
            # define standard draft gear
            spec_i['type'] = 'standard'
            spec_i['Funits'] = 'kips'
            spec_i['Xunits'] = 'in'
            spec_i['xmax'] = np.array([0, 0.47, 0.74, 1.95, 4.94])
            spec_i['Fmax'] = np.array([0, 0, 27, 153, 304])
            spec_i['xmin'] = np.array([0, 0.47, 0.74, 4.7])
            spec_i['Fmin'] = np.array([0, 0, 27, 58 * 1.1])
            spec_i['Klocked'] = (103 - 35) / (1.5 - 1.44) / 4  # Divide by 4 for improved speed
            spec_i['Kmin'] = np.diff(spec_i['Fmin']) / np.diff(spec_i['xmin'])
            spec_i['Kmin'] = np.append(spec_i['Kmin'], spec_i['Kmin'][-1])
            spec_i['Kmax'] = np.diff(spec_i['Fmax']) / np.diff(spec_i['xmax'])
            spec_i['Kmax'] = np.append(spec_i['Kmax'], spec_i['Kmax'][-1])
            spec_i['B'] = 0.2 * 15
            spec_i['Bcushion'] = 0.00001
        
        else:
            # define EOCC coupler - parameterized by preload and stroke
            df0 = 0.015
            df1 = 10 * preload / 100 * stroke / 28
            df3 = 25
            df2 = 1
            df4 = 30
            xdeadzone = 0.43
            K1 = 100 / (0.61 - 0.43)
            x1 = preload / K1 + xdeadzone
            dx1 = (-1.5 + 0.62) * stroke / 28
            assert stroke + dx1 > x1, 'stroke is too small'
            x2 = stroke + dx1
            dx2 = 0.62
            dx3 = 0.68
            
            spec_i['type'] = f'EOCC-{preload}-{stroke}'
            spec_i['Funits'] = 'kips'
            spec_i['Xunits'] = 'in'
            spec_i['xmax'] = np.array([0, xdeadzone, x1, stroke + dx1, stroke + dx2, stroke + dx3])
            spec_i['Fmax'] = np.array([0, df0, preload, preload + df1, preload + df1 + df3, preload + df4 + df1 + df3])
            spec_i['xmin'] = np.array([0, np.nan, np.nan])
            spec_i['Fmin'] = np.array([0, df2, np.nan])
            xmax1 = spec_i['xmax'][-2]
            xmax2 = spec_i['xmax'][-1]
            ymax1 = spec_i['Fmax'][-2]
            ymax2 = spec_i['Fmax'][-1]
            mmin = 0
            mmax = (ymax2 - ymax1) / (xmax2 - xmax1)
            bmin = df2
            bmax = ymax1 - mmax * xmax1
            xintercept = -(bmax - bmin) / (mmax - mmin)
            spec_i['xmin'][1] = xintercept
            spec_i['Fmin'][1] = mmin * xintercept + bmin
            spec_i['xmin'][2] = xmax2
            spec_i['Fmin'][2] = mmax * xmax2 + bmax
            spec_i['Klocked'] = 575  # locked spring constant
            spec_i['Kmin'] = np.diff(spec_i['Fmin']) / np.diff(spec_i['xmin'])
            spec_i['Kmin'] = np.append(spec_i['Kmin'], spec_i['Kmin'][-1])
            spec_i['Kmax'] = np.diff(spec_i['Fmax']) / np.diff(spec_i['xmax'])
            spec_i['Kmax'] = np.append(spec_i['Kmax'], spec_i['Kmax'][-1])
            spec_i['B'] = 0.04 * 7
            spec_i['Bcushion'] = 0.15 * 5 * 100 / preload
        
        spec.append(spec_i)
    
    # find all unique 'x' values of xmin and xmax in the spec for all coupler types
    common = {}
    common['x'] = np.array([])
    for i in range(len(spec)):
        common['x'] = np.concatenate([common['x'], spec[i]['xmax'], spec[i]['xmin']])
    common['x'] = np.unique(common['x'])
    
    # interpolate the coupler forces, Fmax & Fmin, to these unique 'x' values
    # to form coupler force tables that have common 'x' values
    common['Fmax'] = np.zeros((len(spec), len(common['x'])))
    common['Fmin'] = np.zeros((len(spec), len(common['x'])))
    common['Klocked'] = np.zeros(len(spec))
    common['type'] = []
    common['B'] = np.zeros(len(spec))
    common['Bcushion'] = np.zeros(len(spec))
    
    for i in range(len(spec)):
        common['Fmax'][i, :] = interp1d(spec[i]['xmax'], spec[i]['Fmax'], 
                                       kind='linear', fill_value='extrapolate')(common['x'])
        common['Fmin'][i, :] = interp1d(spec[i]['xmin'], spec[i]['Fmin'], 
                                       kind='linear', fill_value='extrapolate')(common['x'])
        common['Klocked'][i] = spec[i]['Klocked']
        common['type'].append(spec[i]['type'])
        common['B'][i] = spec[i]['B']
        common['Bcushion'][i] = spec[i]['Bcushion']
    
    common['Funits'] = spec[0]['Funits']
    common['Xunits'] = spec[0]['Xunits']
    
    funits_list = [s['Funits'] for s in spec]
    if len(set(funits_list)) > 1:
        raise ValueError('Force units not common')
    
    xunits_list = [s['Xunits'] for s in spec]
    if len(set(xunits_list)) > 1:
        raise ValueError('Displacment units not common')
    
    # now find all unique values of force for all coupler types
    inverse = {}
    inverse['F'] = np.array([])
    for i in range(len(spec)):
        inverse['F'] = np.concatenate([inverse['F'], spec[i]['Fmax'], spec[i]['Fmin']])
    inverse['F'] = np.unique(inverse['F'])
    
    # interpolate displacement, xmin & xmax, for each unique value of coupler
    # force (inverse = lookup displacement as a function of force)
    inverse['xmax'] = np.zeros((len(spec), len(inverse['F'])))
    inverse['xmin'] = np.zeros((len(spec), len(inverse['F'])))
    inverse['Klocked'] = np.zeros(len(spec))
    inverse['type'] = []
    
    for i in range(len(spec)):
        # Add small eps values to avoid duplicate x values in interpolation
        Fmax_eps = spec[i]['Fmax'] + np.arange(len(spec[i]['Fmax'])) * np.finfo(float).eps
        Fmin_eps = spec[i]['Fmin'] + np.arange(len(spec[i]['Fmin'])) * np.finfo(float).eps
        
        inverse['xmax'][i, :] = interp1d(Fmax_eps, spec[i]['xmax'], 
                                        kind='linear', fill_value='extrapolate')(inverse['F'])
        inverse['xmin'][i, :] = interp1d(Fmin_eps, spec[i]['xmin'], 
                                        kind='linear', fill_value='extrapolate')(inverse['F'])
        inverse['Klocked'][i] = spec[i]['Klocked']
        inverse['type'].append(spec[i]['type'])
    
    inverse['Funits'] = spec[0]['Funits']
    inverse['Xunits'] = spec[0]['Xunits']
    
    # merge grouped couplers into one assuming that the coupler force is equal
    # for all couplers in a group (sum displacement of inverse x vs F table)
    grouped = {}
    grouped['F'] = inverse['F']
    grouped['xmax'] = np.zeros((numTypesInGroup.shape[0], len(inverse['F'])))
    grouped['xmin'] = np.zeros((numTypesInGroup.shape[0], len(inverse['F'])))
    grouped['Klocked'] = np.zeros(numTypesInGroup.shape[0])
    grouped['B'] = np.zeros(numTypesInGroup.shape[0])
    grouped['Bcushion'] = np.zeros(numTypesInGroup.shape[0])
    
    for i in range(numTypesInGroup.shape[0]):
        grouped['xmax'][i, :] = np.sum(numTypesInGroup[i, :, np.newaxis] * inverse['xmax'], axis=0)
        grouped['xmin'][i, :] = np.sum(numTypesInGroup[i, :, np.newaxis] * inverse['xmin'], axis=0)
        grouped['Klocked'][i] = 1 / np.sum(numTypesInGroup[i, :] / inverse['Klocked'])
        grouped['B'][i] = 1 / np.sum(numTypesInGroup[i, :] / common['B'])
        grouped['Bcushion'][i] = 1 / np.sum(numTypesInGroup[i, :] / common['Bcushion'])
    
    grouped['types'] = inverse['type']
    grouped['ntype'] = numTypesInGroup
    grouped['Funits'] = inverse['Funits']
    grouped['Xunits'] = inverse['Xunits']
    
    # create output structure
    coupler = {
        'spec': spec,
        'common': common,
        'inverse': inverse,
        'grouped': grouped
    }
    
    return coupler