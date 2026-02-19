import numpy as np
from scipy.interpolate import interp1d

def GetESpec_braking(TrainParams, LocoParams, IPOPTParams):
    """Function to get an "expanded spec" from a TrainParams and LocoParams
    structure. This expanded 
    spec is used only within IPOPT and is used to get loco, train and track
    parameters for easy access to the optimizer. This program will be
    modified whenever the spec structure changes
    
    Inputs: TrainParams (Weight, Davis_a, Davis_b, Davis_c), 
            LocoParams: LocoParams structure has the fields Notch, FuelRate, 
                       THP, TracEff, MaxTE (as in the V0 version of the
                       planner). 
            
            IPOPTParams is the Ipopt parameter structure (given by
            GetHCBoundsDef_braking.m, for example) 
    
    Outputs: ESPEC, with the following fields
      1. Loco (1x1 structure)
      2. Train (1x1 structure)
     
    Loco has the following fields:
     1. THP  (horsepower in various notches)
     2. FuelRate (Corresponding vector of fuel rates)
     3. Notch    (Corresponding vector of notches)
     4. MaxTE    (Corresponding effort vector)
     5. Slopes   (Slopes of constraining lines in (alpha, Effort)
        space). In klb-mph 
     6. Intercepts (Intercepts of constraining lines in (alpha,
        Effort) space). In klb.
     7. Hard    (Vector denoting whether each line is hard (<=) or
                 soft (>=) constraint). It is assumed that hard
                 constraints represent motoring effort and soft
                 constraints represent dynamic brake efforts.
     8. MaxTESoft   (Max motoring effort requested - units in klb)
     9. MaxTEHard   (Max motoring effort available - units in klb)
     10. MaxBESoft  (Max dynamic brake effort available - units in klb)
     11.MaxBEHard  (Max total brake effort (DB + air) available - units in
                    klb)
    
    Train has the fields Weight, Davis_a, Davis_b, Davis_c
    """
    
    ###########################################################################
    # Parameters
    
    # small value of slope in (alpha, effort) space, used for finding
    # horizontal line in the limit polygon. This is in pounds-mph
    small_deff_dalpha = 50.0
    
    # Small value of effort, in klb. Efforts smaller than this are
    # negligible. 
    smalleffrt = 0.5
    ###########################################################################
    
    espec = {}
    
    # As in pre-prod, set espec.Train = TrainParams
    espec['Train'] = TrainParams
    
    # CASE 1: USE_SIMPLE_DBMODELS == 1. THIS IS THE CASE THAT NEEDS TO BE
    # CODED IN THE CMU.  
    ############################################################################
    # THE FOLLOWING IS VERY SIMILAR TO GETESPEC IN PRE_PROD, JUST SIMPLER
    ############################################################################  
    n = len(LocoParams)
    for i in range(n):
        # NOTE: Notch goes from 8 to -8. There are no -20, 20 entries
        # in the notch vector (unlike pre-prod).
        if len(LocoParams[i]['THP']) == 9:
            # Flip half of THP, TracEff and MaxTE
            LocoParams[i]['THP'] = np.concatenate([LocoParams[i]['THP'], 
                                                  -np.flip(LocoParams[i]['THP'][:-1])])
        
        if len(LocoParams[i]['TracEff']) == 9:
            # Flip and replicate
            LocoParams[i]['TracEff'] = np.concatenate([LocoParams[i]['TracEff'], 
                                                      np.flip(LocoParams[i]['TracEff'][:-1])])
        
        if len(LocoParams[i]['MaxTE']) == 9:
            LocoParams[i]['MaxTE'] = np.concatenate([LocoParams[i]['MaxTE'], 
                                                    -np.flip(LocoParams[i]['MaxTE'][:-1])])
    
    espec['Loco'] = {}
    
    # Assume that notch vector is common for all locos
    espec['Loco']['Notch'] = LocoParams[0]['Notch']
    espec['Loco']['FuelRate'] = LocoParams[0]['FuelRate']
    # New for V2: Multiply by TracEff right here
    # Convert to numpy arrays for multiplication
    espec['Loco']['THP'] = np.array(LocoParams[0]['THP']) * np.array(LocoParams[0]['TracEff'])
    espec['Loco']['MaxTE'] = LocoParams[0]['MaxTE']
    
    for i in range(1, n):
        # NOTE: Need to take into account active locos here (similar to pre-prod)
        espec['Loco']['FuelRate'] = np.array(espec['Loco']['FuelRate']) + np.array(LocoParams[i]['FuelRate'])
        espec['Loco']['THP'] = espec['Loco']['THP'] + np.array(LocoParams[i]['THP']) * np.array(LocoParams[i]['TracEff'])
        espec['Loco']['MaxTE'] = np.array(espec['Loco']['MaxTE']) + np.array(LocoParams[i]['MaxTE'])
    
    # Flip everything the right way so that notch and THP increase
    espec['Loco']['Notch'] = np.flip(espec['Loco']['Notch'])
    espec['Loco']['MaxTE'] = np.flip(espec['Loco']['MaxTE'])
    espec['Loco']['THP'] = np.flip(espec['Loco']['THP'])
    espec['Loco']['FuelRate'] = np.flip(espec['Loco']['FuelRate'])
    
    ############################################################################
    # NEW FOR V2: COMPUTE SLOPES AND INTERCEPTS FOR THE AGGREGATED SINGLE
    # LOCO 
    ############################################################################ 
    # Traction model
    # Number 375 converts units from horsepower to lbf-mph
    TractionModel = {}
    TractionModel['Slopes'] = np.array([0, espec['Loco']['THP'][-1] * 375])
    TractionModel['Intercepts'] = np.array([espec['Loco']['MaxTE'][-1], 0])
    
    # Braking model
    # Number 375 converts units from horsepower to lbf-mph
    BrakingModel = {}
    BrakingModel['Slopes'] = np.array([0, abs(espec['Loco']['THP'][0] * 375)])
    BrakingModel['Intercepts'] = np.array([abs(espec['Loco']['MaxTE'][0]), 0])
    
    ############################################################################
    # NEW FOR V2: WORK OUT HARD AND SOFT CONSTRAINTS ON TE AND BE. 
    # THIS NEEDS TO BE IN THE CMU VERSION
    ############################################################################  
    # Traction model
    te_slopes = TractionModel['Slopes']
    te_intercepts = TractionModel['Intercepts']
    zero_slope_indices_te = np.where(np.abs(te_slopes) < small_deff_dalpha)[0]
    nonzero_slope_indices_te = np.where(np.abs(te_slopes) >= small_deff_dalpha)[0]
    te_slopes_nz = te_slopes[nonzero_slope_indices_te]
    te_intercepts_nz = te_intercepts[nonzero_slope_indices_te]
    if len(zero_slope_indices_te) > 0:
        te_maxbounds = te_intercepts[zero_slope_indices_te]
        loco_limit = np.min(te_maxbounds / 1000)
    else:
        loco_limit = IPOPTParams['ME']
    
    # motoring constraints need to be handled. If the upper limit on
    # motoring effort is hard, or if the requested maximum motoring
    # effort is greater than the physical limits imposed by the
    # locmotive, we set espec.Loco.MaxTEHard to the locomotive limit,
    # and we set the soft limit to NaN. Otherwise, we 
    # set the hard limit equal to the locomotive limits and the soft
    # limit equal to the requested limit
    if (IPOPTParams['MaxTEConsType'] == 'hard') or (IPOPTParams['ME'] > loco_limit - smalleffrt):
        espec['Loco']['MaxTEHard'] = min(loco_limit, IPOPTParams['ME'])
        espec['Loco']['MaxTESoft'] = np.nan
    else:
        espec['Loco']['MaxTEHard'] = loco_limit
        espec['Loco']['MaxTESoft'] = IPOPTParams['ME']
    
    # Convert everything in terms of klb
    te_slopes_nz = te_slopes_nz / 1000
    te_intercepts_nz = te_intercepts_nz / 1000
    
    # Braking model
    be_slopes = BrakingModel['Slopes']
    be_intercepts = BrakingModel['Intercepts']
    zero_slope_indices_be = np.where(np.abs(be_slopes) < small_deff_dalpha)[0]
    nonzero_slope_indices_be = np.where(np.abs(be_slopes) >= small_deff_dalpha)[0]
    be_slopes_nz = be_slopes[nonzero_slope_indices_be]
    be_intercepts_nz = be_intercepts[nonzero_slope_indices_be]
    if len(zero_slope_indices_be) > 0:
        be_maxbounds = be_intercepts[zero_slope_indices_be]
        espec['Loco']['MaxBESoft'] = min(np.min(be_maxbounds / 1000), 
                                        IPOPTParams['DBE'])
        espec['Loco']['MaxBEHard'] = IPOPTParams['ABE'] + espec['Loco']['MaxBESoft']
    else:
        espec['Loco']['MaxBESoft'] = IPOPTParams['DBE']
        espec['Loco']['MaxBEHard'] = IPOPTParams['ABE'] + espec['Loco']['MaxBESoft']
    
    # Convert everything in terms of klb, change sign since BE is
    # negative 
    be_slopes_nz = -be_slopes_nz / 1000
    be_intercepts_nz = -be_intercepts_nz / 1000
    
    # concatenate slopes and intercepts from TE and BE
    espec['Loco']['Slopes'] = np.concatenate([te_slopes_nz.flatten(), be_slopes_nz.flatten()])
    espec['Loco']['Intercepts'] = np.concatenate([te_intercepts_nz.flatten(), be_intercepts_nz.flatten()])
    espec['Loco']['Hard'] = np.concatenate([np.ones(te_intercepts_nz.size), 
                                           np.zeros(be_intercepts_nz.size)])
    
    # Compute overrate factor for TE
    Pmax = interp1d(espec['Loco']['Notch'], espec['Loco']['THP'], 
                   kind='linear', fill_value='extrapolate')(8.0)
    POverRate = interp1d(espec['Loco']['Notch'], espec['Loco']['THP'], 
                        kind='linear', fill_value='extrapolate')(IPOPTParams['NotchOverRate'])
    espec['Loco']['TEOverRate'] = POverRate / Pmax
    
    return espec