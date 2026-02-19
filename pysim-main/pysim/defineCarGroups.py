# define the car groups that will be lumped together as one rigid car for
# simulation by FastSim
#
# Usage: [cargroup] = defineCarGroups(dcar,couplerTypeIndex,['-verbose','-equalizeSize'])
#
# dcar 3 options
#   dcar (scalar)               grouping independent of coupler type
#   dcar (2 field structure)    group eocc separately from normal couplers
#                               dcar.eocc = # eocc cars in group
#                               dcar.norm = # norm cars in group
#   dcar (3 field structure)    group eocc with different preloads
#                               separately from each other and normal
#                               couplers
#                               dcar.eocc100 = # 100 kip preload eocc cars in group
#                               dcar.eocc050 = #  50 kip preload eocc cars in group
#                               dcar.norm    = # norm cars in group
# couplerTypeIndex              index of coupler types of each vehicle in
#                               the train, 1=norm, 2=eocc 100 kip, 3=eocc 50 kip
# cargroup                      vector of last vehicle in each group
# switches                      -verbose = print results to console
#                               -equalizeSize = adjust groups to be close
#                               to equal size (within coupler type)
#
# H. Kirk Mathews (GE Research) 15 June 2020

import numpy as np

def defineCarGroups(dcar, couplerTypeIndex, *varargin):
    
    # Simple args2options replacement
    defaults = {'equalizeSize': False, 'verbose': False}
    options = defaults.copy()
    
    # Parse varargin
    for arg in varargin:
        if arg == '-verbose':
            options['verbose'] = True
        elif arg == '-equalizeSize':
            options['equalizeSize'] = True
    
    if dcar is None:
        dcar = {'norm': 15, 'eocc': 5}
    
    if isinstance(dcar, dict):
        if not (('eocc' in dcar and 'norm' in dcar) or 
                ('eocc100' in dcar and 'eocc050' in dcar and 'norm' in dcar)):
            fn = list(dcar.keys())
            raise ValueError(f"dcar: unexpected fields {','.join(fn)}")
    elif isinstance(dcar, (int, float)):
        # do nothing to maintain backward compatibility
        pass
    else:
        raise ValueError(f"dcar: unknown type, {type(dcar)}")
    
    n0 = len(couplerTypeIndex)
    
    # Calculate car grouping set
    k = 1
    kg = 1
    cargroup = [1]
    
    if isinstance(dcar, dict) and 'eocc100' in dcar:
        dcarVec = [dcar['norm'], dcar['eocc100'], dcar['eocc050']]
        ctyp = np.array(couplerTypeIndex)
    elif isinstance(dcar, dict) and 'eocc' in dcar:
        dcarVec = [dcar['norm'], dcar['eocc']]
        ctyp = np.array(couplerTypeIndex)
        ctyp[ctyp > 1] = 2
    elif not isinstance(dcar, dict):
        dcarVec = dcar
        ctyp = np.array(couplerTypeIndex)
        ctyp[ctyp > 0] = 1
    
    for i in range(1, n0):
        # Handle scalar dcar case
        if isinstance(dcar, (int, float)):
            group_size_limit = dcar
        else:
            group_size_limit = dcarVec[ctyp[i-1] - 1]
            
        if k + 1 > group_size_limit or ctyp[i-1] != ctyp[i]:
            kg = kg + 1
            k = 0
        k = k + 1
        if kg > len(cargroup):
            cargroup.append(i + 1)
        else:
            cargroup[kg - 1] = i + 1
    
    # Ensure cargroup has the right length
    cargroup = cargroup[:kg]
    
    if options['equalizeSize']:
        cargroup_equalized = cargroup.copy()
        cg0 = [0] + cargroup
        i = 0
        while i < len(cargroup) - 1:
            ig1 = cargroup[i]
            i1 = i
            # Find next different coupler type
            i2 = None
            for j in range(i + 1, len(cargroup)):
                if ctyp[cargroup[j] - 1] != ctyp[cargroup[i1] - 1]:
                    i2 = j - 1
                    break
            if i2 is None:
                i2 = len(cargroup) - 1
            
            ig2 = cargroup[i2]
            tmp = np.round(np.linspace(cg0[i1], cg0[i2 + 1], i2 - i1 + 2)).astype(int)
            for j in range(i1, i2):
                cargroup_equalized[j] = tmp[j - i1 + 1]
            i = i2 + 1
        
        cargroup = cargroup_equalized
    
    if options['verbose']:
        print(''.join([str(x) for x in couplerTypeIndex]))
        s = [' '] * len(ctyp)
        for cg in cargroup:
            s[cg - 1] = '|'
        print(''.join(s))
    
    return cargroup