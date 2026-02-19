# Main simulation pipeline function that orchestrates the complete FastSim workflow.
# This function calls the following sequence:
# 1. runFastSimM - Main simulation with grouped cars
# 2. interpFastSim - Interpolate grouped results to all individual cars
# 3. calcLVratio - Calculate lateral-to-vertical force ratio
# 4. downSampleFastSim - Reduce output data size with intelligent sampling
#
# Usage: fs = runSim(profile, spec, TrackData, Train, Locos, dcar, Ts, dx, range_, X0)
#
# Inputs:
#   profile   - Standard profile structure with time, distance, and operational data
#   spec      - Standard spec structure defining the train and trip
#   TrackData - Track geometry and infrastructure data
#   Train     - Train configuration and parameters
#   Locos     - Locomotive specifications and capabilities
#   dcar      - Number of cars grouped to make FastSim fast (default: None)
#   Ts        - Sample time to provide outputs (default: None)
#   dx        - Distance step size to interpolate profile in miles (default: None)
#   range_    - Distance range to simulate [startDist, endDist] (default: None)
#   X0        - Initial states ('zero', 'steady-state', or array) (default: None)
#
# Output:
#   fs - Complete FastSim output structure with all computed fields including:
#        - Forces for both grouped cars and individual cars
#        - L/V ratios for derailment analysis
#        - Down-sampled data for efficient storage/analysis
#        - All intermediate results and metadata
#
# H. Kirk Mathews (GE Research) - Original MATLAB implementation
# Adapted to Python 2025

import numpy as np
from .runFastSimM import runFastSimM
from .interpFastSim import interpFastSim
from .calcLVratio import calcLVratio
from .downSampleFastSim import downSampleFastSim
from .unit_conversions import lbf_

def runSim(profile, spec, TrackData, Train, Locos, dcar=None, Ts=None, dx=None, range_=None, X0=None):
    """
    Complete train simulation pipeline combining FastSim dynamics with force analysis.
    
    This is the main entry point for running complete train simulations. It orchestrates
    the entire workflow from initial dynamics simulation through final data processing.
    
    Parameters
    ----------
    profile : dict
        Profile structure containing operational data (speed, time, distance, etc.)
    spec : dict
        Specification structure defining train consist and trip parameters
    TrackData : dict
        Track geometry and infrastructure information
    Train : dict
        Train configuration including car weights, lengths, positions
    Locos : list
        Locomotive specifications and performance characteristics
    dcar : int, optional
        Number of cars to group for computational efficiency (default: auto-select)
    Ts : float, optional
        Output sample time in seconds (default: auto-select)
    dx : float, optional
        Distance interpolation step in miles (default: 0.1)
    range_ : list, optional
        Distance range [start, end] in miles (default: full profile)
    X0 : str or array, optional
        Initial conditions ('steady-state', 'zero', or state vector)
        
    Returns
    -------
    dict
        Complete simulation results including:
        - Fsi_allcars: Coupler forces for all individual cars
        - LVratio: Lateral-to-vertical force ratios
        - All grouped and interpolated results
        - Simulation metadata and parameters
    """
    
    # Step 1: Run the main FastSim dynamics simulation with car grouping
    print("Step 1/4: Running FastSim dynamics simulation...")
    fs = runFastSimM(profile, spec, TrackData, Train, Locos, dcar, Ts, dx, range_, X0)
    
    # Step 2: Interpolate grouped car results to individual car forces
    print("Step 2/4: Interpolating results to individual cars...")
    fs = interpFastSim(fs, spec, TrackData, Locos)
    
    # Step 3: Add spec to the output structure for downstream processing
    fs['spec'] = spec
    
    # Step 4: Calculate lateral-to-vertical force ratios for derailment analysis
    print("Step 3/4: Calculating L/V force ratios...")
    # Convert forces from Newtons to pounds for calcLVratio (expects pounds)
    Flb = fs['Fsi_allcars'] / lbf_()  # Convert N to lbf
    fs['LVratio'] = calcLVratio(spec, TrackData, Locos, Flb, fs['dist'])
    
    # Step 5: Down-sample the results to reduce data size while preserving peaks
    print("Step 4/4: Down-sampling results for efficient storage...")
    # Use TFSsave from spec if available, otherwise use a reasonable default
    if 'TFSsave' in spec:
        TFSsave = spec['TFSsave']
    else:
        # Default to 10 seconds if not specified
        TFSsave = 10.0
        print(f"  Using default TFSsave = {TFSsave} seconds")
    
    fs = downSampleFastSim(fs, TFSsave)
    
    print("Simulation pipeline completed successfully!")
    print(f"  Simulation time: {fs['dtsim']:.2f} seconds")
    print(f"  Output time points: {len(fs['t'])}")
    print(f"  Distance range: {fs['dist'][0]:.1f} to {fs['dist'][-1]:.1f} miles")
    
    return fs

def main():
    """
    Example usage and testing function.
    This demonstrates how to use runSim with typical parameters.
    """
    print("runSim.py - Complete Train Simulation Pipeline")
    print("=" * 50)
    print("This module provides the main runSim() function for complete train simulations.")
    print("Usage example:")
    print("  from runSim import runSim")
    print("  fs = runSim(profile, spec, TrackData, Train, Locos)")
    print("")
    print("The function orchestrates the complete simulation pipeline:")
    print("  1. runFastSimM   - Main dynamics simulation")
    print("  2. interpFastSim - Interpolate to individual cars")
    print("  3. calcLVratio   - Calculate L/V force ratios")
    print("  4. downSampleFastSim - Intelligently reduce data size")
    print("")
    print("For detailed usage, see the function docstring or test scripts.")

if __name__ == "__main__":
    main()