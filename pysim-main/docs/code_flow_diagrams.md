# FastSim Code Flow Analysis

## Call Graph

```
danFastSim.m (Entry Point)
│
├── load('danFastSimSampleInputs.mat')  // Load input data
│
├── runFastSimM()  // Main simulation orchestrator
│   │
│   ├── normalizeProfile()  // Interpolate track profile to fixed steps
│   │   └── upsampleProfile()  // Profile interpolation utility
│   │
│   ├── calcRopeForces()  // Calculate locomotive forces
│   │   ├── getfulltrackstruc()  // Load track database
│   │   ├── getlocostruc()  // Load locomotive parameters
│   │   ├── gravityforces()  // Calculate grade forces
│   │   ├── locoeffort_of_notch_speed()  // Locomotive traction curves
│   │   └── slewRateFilter()  // Apply throttle rate limits (optional)
│   │
│   ├── **UNIT CONVERSION**: Frope_in0 = FropeFunction*lbf_  // Convert to SI (Newtons)
│   │
│   ├── initFastSim()  // Initialize simulation parameters
│   │   ├── optspec_dtm()  // Apply DTM (Detailed Train Makeup) logic
│   │   ├── getlocostruc()  // Get locomotive specifications
│   │   ├── defineCarGroups()  // Group cars for computational efficiency
│   │   │   └── args2options()  // Parse function arguments
│   │   ├── **UNIT CONVERSION**: groupMassVector(i) = sum(weights)*ton_  // Convert to SI (kg)
│   │   ├── **UNIT CONVERSION**: groupDampingB = coupler.B*kips_/(in_/sec_)  // Convert to SI
│   │   ├── defineCouplers()  // Define coupler force-displacement tables
│   │   └── ssCouplerDisplacement()  // Calculate steady-state displacements
│   │
│   ├── FastSimM()  // Core physics simulation (ODE integration)
│   │   ├── ssCouplerDisplacement()  // Initial state calculation
│   │   └── ode23simple()  // Adaptive step-size ODE solver
│   │       └── FastSimODE()  // Physics equations (called repeatedly) **ALL IN SI UNITS**
│   │           ├── // Interpolate rope forces at current time (Newtons)
│   │           ├── // Calculate coupler forces with hysteresis (Newtons)
│   │           ├── // Apply force limits: Fmin ≤ F ≤ Fmax (Newtons)
│   │           ├── // Compute car accelerations: F=ma (kg, m/s²)
│   │           └── // Return state derivatives [F̈, V̇, Ẋ] (SI units)
│   │
│   ├── **UNIT CONVERSION**: dist = interp1(profile.Time*hour_, profile.Dist, t*sec_)
│   └── // Post-process and format results (SI units)
│
├── interpFastSim()  // Interpolate group forces to individual cars
│   └── calcRopeForces()  // Recalculate rope forces for all cars
│
├── **UNIT CONVERSION**: fs.LVratio = calcLVratio(spec1, fs.Fsi_allcars/lbf_, fs.dist)
├── calcLVratio()  // Calculate L/V (Lateral/Vertical) force ratios
│
└── downSampleFastSim()  // Reduce output data size
    └── // Sign-aware moving max to preserve peak forces

```

## Sequence Diagram

```
User → danFastSim: Start simulation
danFastSim → MATLAB: load('danFastSimSampleInputs.mat')
MATLAB → danFastSim: {Profile1, spec1, dcar, Ts, dx, X0}

danFastSim → runFastSimM: (Profile1, spec1, dcar, Ts, dx, range, X0)

    runFastSimM → normalizeProfile: (profile, spec, dx)
        normalizeProfile → upsampleProfile: (spec, profile, dist_pattern)
        upsampleProfile → normalizeProfile: interpolated_profile
    normalizeProfile → runFastSimM: normalized_profile

    runFastSimM → calcRopeForces: (profile, spec)
        calcRopeForces → getfulltrackstruc: (spec.Route)
        getfulltrackstruc → calcRopeForces: track_database
        
        calcRopeForces → getlocostruc: (spec.Consist)
        getlocostruc → calcRopeForces: loco_parameters
        
        calcRopeForces → gravityforces: (profile, train_params)
        gravityforces → calcRopeForces: grade_forces
        
        calcRopeForces → locoeffort_of_notch_speed: (notch, speed, loco_params)
        locoeffort_of_notch_speed → calcRopeForces: traction_forces
        
        calcRopeForces → slewRateFilter: (forces) [optional]
        slewRateFilter → calcRopeForces: rate_limited_forces
    calcRopeForces → runFastSimM: rope_forces_function

    runFastSimM → initFastSim: (spec, dcar, Ts)
        initFastSim → optspec_dtm: (spec)
        optspec_dtm → initFastSim: processed_spec
        
        initFastSim → getlocostruc: (spec.Consist)
        getlocostruc → initFastSim: loco_specs
        
        initFastSim → defineCarGroups: (dcar, coupler_types)
        defineCarGroups → initFastSim: car_groups
        
        initFastSim → defineCouplers: (coupler_specs)
        defineCouplers → initFastSim: {x2all, Fmin2all, Fmax2all}
        
        initFastSim → ssCouplerDisplacement: (x2all, Fmin2all, Fmax2all, initial_forces)
        ssCouplerDisplacement → initFastSim: initial_displacements
    initFastSim → runFastSimM: {Klocked, Kawu, masses, damping, car_groups, force_tables}

    runFastSimM → FastSimM: (time_vector, rope_forces, sim_params, initial_state)
        FastSimM → ssCouplerDisplacement: (force_tables, initial_rope_forces) [if X0='steady-state']
        ssCouplerDisplacement → FastSimM: initial_state_vector
        
        FastSimM → ode23simple: (ode_function, time_span, initial_state, options)
            
            loop [Adaptive time stepping]
                ode23simple → FastSimODE: (t, X, simulation_parameters)
                    FastSimODE → FastSimODE: interpolate_rope_forces(t)
                    FastSimODE → FastSimODE: lookup_force_limits(displacement)
                    FastSimODE → FastSimODE: apply_hysteresis_limits(forces)
                    FastSimODE → FastSimODE: calculate_car_accelerations()
                    FastSimODE → FastSimODE: compute_state_derivatives()
                FastSimODE → ode23simple: [Ẋdot, rope_forces, coupler_forces, Fmin, Fmax]
                
                ode23simple → ode23simple: estimate_error()
                ode23simple → ode23simple: adapt_step_size()
            end
            
        ode23simple → FastSimM: {time_out, states_out, stats}
    FastSimM → runFastSimM: simulation_results

    runFastSimM → runFastSimM: post_process_results()
runFastSimM → danFastSim: FastSim_output_structure

danFastSim → interpFastSim: (fs, spec1)
    interpFastSim → calcRopeForces: (profile, spec) [for all individual cars]
    calcRopeForces → interpFastSim: individual_car_rope_forces
    interpFastSim → interpFastSim: interpolate_group_forces_to_individual_cars()
interpFastSim → danFastSim: fs_with_individual_car_forces

danFastSim → calcLVratio: (spec1, forces, distances)
calcLVratio → danFastSim: L_V_ratios

danFastSim → downSampleFastSim: (fs, spec1.TFSsave)
    downSampleFastSim → downSampleFastSim: sign_aware_moving_max()
downSampleFastSim → danFastSim: downsampled_results

danFastSim → User: Final simulation results
```

## Key Data Flow

### Input Data Structure:
```
Profile1: {
    Dist: [98.0 → 259.0 miles, 929 points]
    Time: [0 → 4.0 hours]
    Speed: [22.8 → 70.0 mph] 
    Frope: [136 couplers × 929 distance points]
}

spec1: {
    Train: {
        NumCars: 132
        Length: 7484 ft
        LoadWeight: [132 car weights in tons]
        LocoPosition: [1, 2, 135, 136]
    }
    Route: Track database reference
    Trip: {Speed limits, start/end points, PLM commands}
}
```

### Simulation Parameters:
- `Ts = 0.5` seconds (output sample time)
- `dcar = 5` cars per group (computational efficiency)
- `dx = 0.1` miles (profile interpolation step)

### Core Physics Loop:
1. **Time Integration**: `ode23simple` adaptively steps through time
2. **At each time step**: `FastSimODE` calculates:
   - Current rope forces (from locomotive)
   - Coupler forces with hysteresis limits
   - Car accelerations using Newton's laws
   - State derivatives for next integration step

### Output Processing:
1. **Group → Individual**: `interpFastSim` expands group results to individual cars
2. **Safety Analysis**: `calcLVratio` computes derailment risk metrics  
3. **Data Reduction**: `downSampleFastSim` reduces file size while preserving peaks

## Unit Conversion Strategy

**Key Insight**: Unit conversions happen at the **boundaries** of the core physics simulation, not inside it!

### **Input Conversion (US → SI)**:
- **Forces**: `lbf_` converts pounds-force to Newtons 
- **Masses**: `ton_` converts US tons to kilograms
- **Damping**: `kips_/(in_/sec_)` converts to SI damping units
- **Time**: Input profile times converted from hours to seconds

### **Core Physics (Pure SI)**:
- `FastSimODE()` operates entirely in SI units (N, kg, m, s)
- No unit conversions inside the physics loops
- Maintains computational efficiency and numerical consistency

### **Output Conversion (SI → Mixed)**:
- **Time interpolation**: `hour_` and `sec_` for profile mapping
- **Force reporting**: `/lbf_` converts Newtons back to pounds-force for L/V analysis
- **Results structure**: Maintains SI internally, converts on demand

### **Unit Functions Called**:
```
Input Stage:    lbf_(), ton_(), kips_(), in_(), sec_()
Core Physics:   (None - pure SI)
Output Stage:   hour_(), sec_(), lbf_() [for reporting]
```

## Computational Strategy

The system uses **car grouping** to balance accuracy vs. speed:
- 132 cars → ~28 groups of 5 cars each
- Simulates forces only at group boundaries
- Post-processes to interpolate forces within groups
- Reduces ODE system size by ~80% while maintaining essential physics