def GetHCBoundsDef_braking():
    """Get the global (hardcoded) bounds imposed on all quantities. Also get
    defaults for algorithm parameters"""
    
    HCBoundsDef = {}
    HCBoundsDef['Bounds'] = {}
    HCBoundsDef['Def'] = {}
    
    # GLOBAL lower and upper bounds on any speed. mph
    HCBoundsDef['Bounds']['SpdBound'] = [5.0, 75.0]
    # Lower bound on separation between any two points xi and xj (mi) in the
    # track database-used for error checking
    HCBoundsDef['Bounds']['SepDist'] = 0.0001
    # Bounds on mesh diameter. The upper and lower bounds on mesh dia must be
    # consistent with the bound on separation distance
    HCBoundsDef['Bounds']['MeshDia'] = [0.01, 0.25]
    # Bounds on segment length - NOTE! THE LOWER BOUND MUST BE AT LEAST 3
    # TIMES THE LOWER BOUND ON MESH DIA!!!!
    HCBoundsDef['Bounds']['SegLen'] = [1, 2000]
    # Bounds on max trip time, derived from min speed and max segment length,
    # and max speed and min segment length
    HCBoundsDef['Bounds']['MaxTripTime'] = [0.001, 1000]
    # Lower and upper limits on absolute value of Effort rate limit
    HCBoundsDef['Bounds']['EffortRateLim'] = [1, 20]  # klb per s
    # Bounds for any notch that appears in the inputs (such as start and
    # end notches)
    HCBoundsDef['Bounds']['NotchLim'] = [-8, 8]
    # Bound on absolute total motoring effort (klb)
    HCBoundsDef['Bounds']['ME'] = [50, 1000]
    # Bound on absolute total dynamic braking effort (klb)
    HCBoundsDef['Bounds']['DBE'] = [50, 1000]
    # Bound on absolute total air braking effort (klb)
    HCBoundsDef['Bounds']['ABE'] = [100, 5000]
    # AirBrakeWarningLevel. If DB requirement is equal to max value -
    # AirBrakeWarningLevel, print a warning that air brake may be required. This
    # is in klb
    HCBoundsDef['Bounds']['AirBrakeWarningLevel'] = [0, 50]
    # Bounds on spdlimpointdist (distance from any speed limit within
    # which to use small mesh dia
    HCBoundsDef['Bounds']['SpdLimPointDist'] = [0.1, 0.5]
    # Bounds on spdlimpointMeshDia (Mesh dia used close to speed limits)
    HCBoundsDef['Bounds']['SpdLimPointMeshDia'] = [0.01, 0.25]
    
    # Lower and upper limits on time objective weight
    HCBoundsDef['Bounds']['TimeObjWeight'] = [0.0, 1.0]
    # Lower and upper limits on fuel objective weight
    HCBoundsDef['Bounds']['FuelObjWeight'] = [0.0, 1.0]
    # Lower and upper bounds on fuel-time tradeoff. Units = pounds per 1
    # hour time
    HCBoundsDef['Bounds']['FuelTimeTradeoff'] = [100, 10000]
    # Lower and upper bounds on fuel-THPrate tradeoff. Units = pounds
    # fuel per [(1000 horsepower)^2/hour]. This
    # is normalized by the number of locos.
    HCBoundsDef['Bounds']['FuelTHPRateTradeoff'] = [1e-4, 1]
    # Lower and upper bounds on fuel-speed error tradeoff. Units = pounds
    # fuel per [mph^2]
    HCBoundsDef['Bounds']['FuelSpdTradeoff'] = [0, 5000]
    # Lower and upper bounds on fuel-TE error tradeoff. Units = pounds
    # fuel per klb^2
    HCBoundsDef['Bounds']['FuelEffortTradeoff'] = [0, 1000]
    # Lower and upper bounds on fuel-BE violation tradeoff. Units = pounds
    # fuel per klb. Used to penalize l-infinity norm of air brake use
    HCBoundsDef['Bounds']['FuelABETradeoff'] = [1, 100]
    # fuel-TE limit violation tradeoff. Units = pounds fuel per klb
    HCBoundsDef['Bounds']['FuelTEViolTradeoff'] = [1000, 20000]
    # Lower and upper bounds on fuel-BE rate violation tradeoff. Units = pounds
    # fuel per [klb/s]. Used to penalize l-infinity norm of effort rate violation
    HCBoundsDef['Bounds']['FuelEffortRateTradeoff'] = [1e4, 1e6]
    # Lower and upper limits on HP objective weight
    HCBoundsDef['Bounds']['HPObjWeight'] = [0.0, 1.0]
    # Lower and upper limits on Speed reference objective weight
    HCBoundsDef['Bounds']['SpdRefSoft'] = [0.0, 1.0]
    # Lower and upper limits on Effort reference objective weight
    HCBoundsDef['Bounds']['EffortRefSoft'] = [0.0, 1.0]
    # Penalty on violation of BE constraint (violation may be necessary
    # in case air brakes are needed)
    HCBoundsDef['Bounds']['BESoft'] = [0.0, 1.0]
    # Penalty on violation of TE constraint (violation may be necessary
    # to get the train over a hill)
    HCBoundsDef['Bounds']['TESoft'] = [0.0, 1.0]
    # Penalty on violation of rate constraint
    HCBoundsDef['Bounds']['EffortRateSoft'] = [0.0, 1.0]
    # Lower and upper limits on Effort normalization (in kilo pounds)
    HCBoundsDef['Bounds']['EffortNorm'] = [5, 500]
    # On fuel objective (pounds)
    HCBoundsDef['Bounds']['FuelNorm'] = [100, 10000]
    # On speed normalization constant
    HCBoundsDef['Bounds']['SpdNorm'] = [5.0, 70.0]
    # Tolerance for ipopt
    HCBoundsDef['Bounds']['Tol'] = [1e-6, 1e-4]
    # Size bounds for tolerance interp. As problem size(number of mesh points)
    # varies between these bounds, tolerance varies between upper and lower
    # bounds. For problem sizes outside this interval, tol is clamped.
    HCBoundsDef['Bounds']['TolProbSize'] = [100, 1000]
    # To over-rate (new term!) notch and avoid notch hill problem, we
    # can tell the loco that there is more TE available by the following
    # factor 
    HCBoundsDef['Bounds']['NotchOverRate'] = [8.0, 10.0]
    
    # Speed below which max notch is allowed
    HCBoundsDef['Bounds']['MaxNotchVThreshold'] = [40, 70]
    # When above MaxNotchVThreshold, what is the max allowed notch?
    HCBoundsDef['Bounds']['MaxNotchHiSpeed'] = [2, 8]
    
    # Max iterations for ipopt
    HCBoundsDef['Bounds']['MaxIter'] = [50, 1000]
    # Kernel index
    HCBoundsDef['Bounds']['Kernel'] = [0, 4]
    
    # Regularization
    #   TimeObjThreshold  : If TimeObjWeight <= TimeObjThreshold, problem is
    #                       NOT considered to be a min-time problem
    HCBoundsDef['Bounds']['TimeObjThreshold'] = [0, 1]
    
    #   TimeFuelRatioThreshold : If TimeObjWeight > TimeObjThreshold and
    #                            TimeObjWeight >
    #                            FuelObjWeight*TimeFuelRatioThreshold, problem
    #                            is recognised as a min-time problem
    HCBoundsDef['Bounds']['TimeFuelRatioThreshold'] = [10, 1000]
    
    #   TripTimeFactor    : If MaxTripTime > TripTimeFactor times the
    #                       time taken to complete the trip at the speed limit, 
    #                       the time objective weight is gradually increased so
    #                       that optimization is not ill-posed.
    HCBoundsDef['Bounds']['TripTimeFactor'] = [1, 10]
    
    #   TripTimeFactorThreshold : If MaxTripTime > TripTimeFactorThreshold
    #                             times the time taken to complete the trip at
    #                             the speed limit, TimeObjWeight is set to
    #                             TimeObjMinThreshold  
    HCBoundsDef['Bounds']['TripTimeFactorThreshold'] = [2, 20]
    HCBoundsDef['Bounds']['TimeObjMinThreshold'] = [0, 1]
    
    # Ruling grade margin. This is added to ruling grade to calculate if the
    # train can get over the steepest hill
    HCBoundsDef['Bounds']['RulingGradeMargin'] = [0, 1.0]
    # Whether or not we want to override the Max TE constraint at low speeds
    HCBoundsDef['Bounds']['MaxTEOverride'] = [0, 1]
    
    ###########################################################################
    # Defaults
    HCBoundsDef['Def']['EffortRateLim'] = [-2, 2]  # klb per sec
    HCBoundsDef['Def']['InteriorMinSpd'] = 5.0  # mph
    HCBoundsDef['Def']['BoundaryMinSpd'] = 5.0  # mph
    HCBoundsDef['Def']['MaxTripTime'] = 1000
    # Bound on absolute total motoring effort (klb)
    HCBoundsDef['Def']['ME'] = 1000
    # Soft Bound on absolute total dynamic braking effort (klb)
    HCBoundsDef['Def']['DBE'] = 1000
    # Bound on absolute total air braking effort (klb)
    HCBoundsDef['Def']['ABE'] = 500
    # AirBrakeWarningLevel. If DB requirement is equal to max value -
    # AirBrakeWarningLevel, print a warning that air brake may be required. This
    # is in klb
    HCBoundsDef['Def']['AirBrakeWarningLevel'] = 10
    
    HCBoundsDef['Def']['TimeObjWeight'] = 1.0
    HCBoundsDef['Def']['FuelObjWeight'] = 1e-4
    HCBoundsDef['Def']['HPObjWeight'] = 0.5
    
    # Regularization
    #   TimeObjThreshold  : If TimeObjWeight <= TimeObjThreshold, problem is
    #                       NOT considered to be a min-time problem
    HCBoundsDef['Def']['TimeObjThreshold'] = 0.5
    
    #   TimeFuelRatioThreshold : If TimeObjWeight > TimeObjThreshold and
    #                            TimeObjWeight >
    #                            FuelObjWeight*TimeFuelRatioThreshold, problem
    #                            is recognised as a min-time problem
    HCBoundsDef['Def']['TimeFuelRatioThreshold'] = 100.0
    
    #   TripTimeFactor    : If MaxTripTime > TripTimeFactor times the
    #                       time taken to complete the trip at the speed limit, 
    #                       the time objective weight is gradually increased so
    #                       that optimization is not ill-posed.
    HCBoundsDef['Def']['TripTimeFactor'] = 5.0
    
    #   TripTimeFactorThreshold : If 
    #                              a) MaxTripTime >= TripTimeFactorThreshold
    #                                 times the time taken to complete the trip at
    #                                 the speed limit, OR 
    #                              b) MaxTripTime >= time taken to complete
    #                                 the trip at the minimum speed, 
    #                             then
    #                              TimeObjWeight is set to TimeObjMinThreshold 
    HCBoundsDef['Def']['TripTimeFactorThreshold'] = 10.0
    HCBoundsDef['Def']['TimeObjMinThreshold'] = 0.1
    # Ruling grade margin. This is added to ruling grade to calculate if the
    # train can get over the steepest hill
    HCBoundsDef['Def']['RulingGradeMargin'] = 0.05
    
    # fuel-time tradeoff. Units = pounds per hour
    HCBoundsDef['Def']['FuelTimeTradeoff'] = 1000
    # fuel-THPrate tradeoff. Units = pound hours per (1000 horsepower)^2. This
    # is normalized by the square of the number of locos.
    HCBoundsDef['Def']['FuelTHPRateTradeoff'] = 4e-2
    # fuel-speed error tradeoff. Units = pounds fuel per [mph^2]
    HCBoundsDef['Def']['FuelSpdTradeoff'] = 1000
    # fuel-TE error tradeoff. Units = pounds fuel per klb^2
    HCBoundsDef['Def']['FuelEffortTradeoff'] = 50
    # fuel-TE limit violation tradeoff. Units = pounds fuel per klb
    HCBoundsDef['Def']['FuelTEViolTradeoff'] = 5000
    # fuel-BE violation tradeoff. Units = pounds fuel per klb. Used to
    # penalize l-infinity norm of air brake use 
    HCBoundsDef['Def']['FuelABETradeoff'] = 20
    # fuel-Effort rate violation tradeoff. Units = pounds fuel per
    # [klb/s].  
    HCBoundsDef['Def']['FuelEffortRateTradeoff'] = 5.0e+005
    
    HCBoundsDef['Def']['BESoft'] = 1.0
    HCBoundsDef['Def']['TESoft'] = 1.0
    HCBoundsDef['Def']['EffortRateSoft'] = 1.0
    HCBoundsDef['Def']['SpdRefSoft'] = 1.0
    HCBoundsDef['Def']['EffortRefSoft'] = 1.0
    
    HCBoundsDef['Def']['EffortNorm'] = 50
    HCBoundsDef['Def']['FuelNorm'] = 5000
    # Harmonic mean of the bounds?
    HCBoundsDef['Def']['SpdNorm'] = 35
    
    HCBoundsDef['Def']['StartNotchConsType'] = 'free'  # 'soft', 'free'
    HCBoundsDef['Def']['EndNotchConsType'] = 'free'  # 'soft', 'free'
    HCBoundsDef['Def']['StartSpdConsType'] = 'soft'  # 'hard', 'soft', 'free'
    HCBoundsDef['Def']['EndSpdConsType'] = 'soft'  # 'hard', 'soft', 'free'
    HCBoundsDef['Def']['MaxTEConsType'] = 'hard'  # 'hard', 'soft'
    
    HCBoundsDef['Def']['StartSpd'] = 10.0  # mph
    HCBoundsDef['Def']['EndSpd'] = 10.0  # mph
    HCBoundsDef['Def']['StartNotch'] = 1.0
    HCBoundsDef['Def']['EndNotch'] = 1.0
    
    HCBoundsDef['Def']['LowAvgSpd'] = 30  # mph, average speed below which to use a dense mesh
    HCBoundsDef['Def']['InteriorMeshDia'] = [0.08, 0.25]  # Mesh dia is adaptively
                                                          # chosen to be
                                                          # approximately between
                                                          # these limits
    HCBoundsDef['Def']['BoundaryMeshDia'] = 0.02  # Mesh dia near boundaries
    # spdlimpointdist (distance from any speed limit within
    # which to use small mesh dia
    HCBoundsDef['Def']['SpdLimPointDist'] = 0.25
    HCBoundsDef['Def']['SpdLimPointMeshDia'] = 0.08
    
    HCBoundsDef['Def']['MaxIter'] = 250
    HCBoundsDef['Def']['Kernel'] = 4
    HCBoundsDef['Def']['Verbosity'] = 0
    # To avoid idle, set this parameter equal to 1
    HCBoundsDef['Def']['AvoidIdle'] = 0
    # To over-rate (new term!) notch and avoid notch hill problem, we
    # can tell the loco that there is more TE available by the following
    # factor 
    HCBoundsDef['Def']['NotchOverRate'] = 8
    
    # To impose a rule whereby one is not allowed to go above notch k
    # when the speed is > v mph, set this parameter equal to 1
    HCBoundsDef['Def']['VelDepNotchConstr'] = 0
    # Speed below which max notch is allowed (ie, the 55 mph in the 5-55 rule)
    HCBoundsDef['Def']['MaxNotchVThreshold'] = 55
    # When above MaxNotchVThreshold, what is the max allowed notch? (ie,
    # notch 5 in the 5-55 rule)
    HCBoundsDef['Def']['MaxNotchHiSpeed'] = 5
    # Want to over-ride max te?
    HCBoundsDef['Def']['MaxTEOverRide'] = 1
    
    return HCBoundsDef