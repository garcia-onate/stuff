#!/usr/bin/env python3
"""
Test script for runFastSimM.py function
This script loads test data and validates the runFastSimM function output.
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runFastSimM import runFastSimM

def load_json_data(filename):
    """Load and parse JSON data from file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def load_csv_data(filename):
    """Load CSV data from file."""
    try:
        data = pd.read_csv(filename, header=None).values.flatten()
        return data
    except Exception as e:
        print(f"Error loading CSV data from {filename}: {e}")
        return None

def validate_data_structures(profile_data, spec_data, track_data, train_data, locos_data, range_data):
    """Validate that all required data structures are present and well-formed."""
    print("Validating data structures...")
    
    # Validate profile data
    if not isinstance(profile_data, dict):
        print("ERROR: Profile data should be a dictionary")
        return False
    
    required_profile_fields = ['Dist', 'Time', 'Speed', 'CNotch']
    for field in required_profile_fields:
        if field not in profile_data:
            print(f"ERROR: Missing required field '{field}' in profile data")
            return False
    
    print(f"Profile data: {len(profile_data['Dist'])} points")
    print(f"  Distance range: {profile_data['Dist'][0]:.1f} to {profile_data['Dist'][-1]:.1f} miles")
    print(f"  Time range: {profile_data['Time'][0]:.2f} to {profile_data['Time'][-1]:.2f} hours")
    
    # Validate spec data
    if not isinstance(spec_data, dict):
        print("ERROR: Spec data should be a dictionary")
        return False
    
    required_spec_fields = ['Train', 'Consist']
    for field in required_spec_fields:
        if field not in spec_data:
            print(f"ERROR: Missing required field '{field}' in spec data")
            return False
    
    # Validate Train data within spec
    train_data_from_spec = spec_data['Train']
    required_train_fields = ['NumCars', 'LocoPosition', 'LoadWeight']
    for field in required_train_fields:
        if field not in train_data_from_spec:
            print(f"ERROR: Missing required field '{field}' in Train data from spec")
            return False
    
    print(f"Spec data: {train_data_from_spec['NumCars']} cars, locomotives at positions {train_data_from_spec['LocoPosition']}")
    
    # Validate TrackData
    if not isinstance(track_data, dict):
        print("ERROR: TrackData should be a dictionary")
        return False
    
    print(f"TrackData contains: {list(track_data.keys())}")
    
    # Validate Train data
    if not isinstance(train_data, dict):
        print("ERROR: Train data should be a dictionary")
        return False
    
    required_train_fields = ['NumCars', 'LocoPosition', 'LoadWeight']
    for field in required_train_fields:
        if field not in train_data:
            print(f"ERROR: Missing required field '{field}' in Train data")
            return False
    
    print(f"Train data: {train_data['NumCars']} cars, locomotives at positions {train_data['LocoPosition']}")
    
    # Validate Locos data
    if not isinstance(locos_data, list) or len(locos_data) == 0:
        print("ERROR: Locos data should be a non-empty list")
        return False
    
    print(f"Locomotive data: {len(locos_data)} locomotives")
    for i, loco in enumerate(locos_data):
        if 'Weight' not in loco:
            print(f"ERROR: Locomotive {i} missing 'Weight' field")
            return False
        print(f"  Loco {i+1}: {loco.get('LocoName', 'Unknown')} - {loco['Weight']} tons")
    
    # Validate range data
    if range_data is None or len(range_data) != 2:
        print("ERROR: Range data should be an array with 2 elements [start, end]")
        return False
    
    print(f"Range data: {range_data[0]} to {range_data[1]} miles")
    
    return True

def run_runFastSimM_test():
    """Main test function for runFastSimM."""
    print("="*60)
    print("TESTING runFastSimM.py")
    print("="*60)
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test data files
    print("\nLoading test data files...")
    
    # Load profile configuration
    profile_file = os.path.join(test_dir, 'profile.txt')
    profile_data = load_json_data(profile_file)
    if profile_data is None:
        print("Failed to load profile data")
        return False
    
    # Load spec configuration
    spec_file = os.path.join(test_dir, 'spec.txt')
    spec_data = load_json_data(spec_file)
    if spec_data is None:
        print("Failed to load spec data")
        return False
    
    # Load TrackData
    track_file = os.path.join(test_dir, 'TrackData.txt')
    track_data = load_json_data(track_file)
    if track_data is None:
        print("Failed to load TrackData")
        return False
    
    # Load Train data
    train_file = os.path.join(test_dir, 'Train.txt')
    train_data = load_json_data(train_file)
    if train_data is None:
        print("Failed to load Train data")
        return False
    
    # Load Locomotives data
    locos_file = os.path.join(test_dir, 'Locos.txt')
    locos_data = load_json_data(locos_file)
    if locos_data is None:
        print("Failed to load Locos data")
        return False
    
    # Load range data
    range_file = os.path.join(test_dir, 'range.csv')
    range_data = load_csv_data(range_file)
    if range_data is None:
        print("Failed to load range data")
        return False
    
    # Validate data structures
    if not validate_data_structures(profile_data, spec_data, track_data, train_data, locos_data, range_data):
        print("Data validation failed")
        return False
    
    print("\n" + "="*60)
    print("RUNNING runFastSimM FUNCTION")
    print("="*60)
    
    # Set test parameters as specified
    dcar = 5
    Ts = 0.5
    dx = 0.1
    X0 = 'steady-state'
    
    print(f"Test parameters:")
    print(f"  dcar (cars per group): {dcar}")
    print(f"  Ts (time step): {Ts}")
    print(f"  dx (distance step): {dx}")
    print(f"  range: [{range_data[0]}, {range_data[1]}]")
    print(f"  X0 (initial state): {X0}")
    
    try:       
        # Call runFastSimM function
        print(f"\nCalling runFastSimM...")
        start_time = time.time()
        
        result = runFastSimM(profile_data, spec_data, track_data, train_data, locos_data, 
                           dcar=dcar, Ts=Ts, dx=dx, range_=range_data, X0=X0)
        
        execution_time = time.time() - start_time
        
        print(f"runFastSimM completed successfully in {execution_time:.2f} seconds!")
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        print(f"Output structure contains {len(result)} fields:")
        for key in sorted(result.keys()):
            value = result[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: array shape {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} length {len(value)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Analyze key results
        print(f"\nSimulation results:")
        print(f"  Time points: {len(result['t'])}")
        print(f"  Time range: {result['t'][0]:.2f} to {result['t'][-1]:.2f} seconds")
        print(f"  Distance range: {result['dist'][0]:.2f} to {result['dist'][-1]:.2f} miles")
        print(f"  Number of car groups: {len(result['cargroup'])}")
        print(f"  Car group boundaries: {result['cargroup']}")
        
        # Analyze forces
        F = result['Fsi']
        F1 = result['F1si']
        Frope = result['Fropesi']
        
        print(f"\nForce analysis:")
        print(f"  Coupler forces (Fsi) shape: {F.shape}")
        print(f"  Force range: {np.min(F):.0f} to {np.max(F):.0f} N")
        print(f"  RMS force: {np.sqrt(np.mean(F**2)):.0f} N")
        
        print(f"  Rope forces (Fropesi) shape: {Frope.shape}")
        print(f"  Rope force range: {np.min(Frope):.0f} to {np.max(Frope):.0f} N")
        
        # Analyze displacements and velocities
        DX = result['DXsi']
        DV = result['DVsi']
        
        print(f"\nDynamic analysis:")
        print(f"  Relative displacements (DXsi) shape: {DX.shape}")
        print(f"  Displacement range: {np.min(DX):.4f} to {np.max(DX):.4f} m")
        print(f"  RMS displacement: {np.sqrt(np.mean(DX**2)):.4f} m")
        
        print(f"  Relative velocities (DVsi) shape: {DV.shape}")
        print(f"  Velocity range: {np.min(DV):.4f} to {np.max(DV):.4f} m/s")
        print(f"  RMS velocity: {np.sqrt(np.mean(DV**2)):.4f} m/s")
        
        # Sanity checks
        print(f"\nSanity checks:")
        
        # Check for reasonable values
        max_force = np.max(np.abs(F))
        if max_force > 1e7:  # 10 MN seems very large
            print(f"  WARNING: Very large forces detected ({max_force:.0f} N)")
        else:
            print(f"  ✓ Maximum force magnitude reasonable ({max_force:.0f} N)")
        
        # Check for NaN values
        arrays_to_check = [F, F1, DX, DV, Frope, result['t'], result['dist']]
        array_names = ['Fsi', 'F1si', 'DXsi', 'DVsi', 'Fropesi', 't', 'dist']
        
        nan_found = False
        for arr, name in zip(arrays_to_check, array_names):
            if np.any(np.isnan(arr)):
                print(f"  ERROR: NaN values detected in {name}")
                nan_found = True
        
        if not nan_found:
            print("  ✓ No NaN values in key results")
        
        # Check for infinite values
        inf_found = False
        for arr, name in zip(arrays_to_check, array_names):
            if np.any(np.isinf(arr)):
                print(f"  ERROR: Infinite values detected in {name}")
                inf_found = True
        
        if not inf_found:
            print("  ✓ No infinite values in key results")
        
        # Check time consistency
        if not np.all(np.diff(result['t']) > 0):
            print("  ERROR: Time vector is not monotonically increasing")
            return False
        else:
            print("  ✓ Time vector is monotonically increasing")
        
        # Check distance consistency
        if not np.all(np.diff(result['dist']) >= 0):
            print("  WARNING: Distance vector is not monotonically increasing")
        else:
            print("  ✓ Distance vector is monotonically increasing")
        
        # Check simulation time
        sim_time = result['dtsim']
        print(f"  Simulation execution time: {sim_time:.3f} seconds")
        if sim_time > 60:  # More than 1 minute seems long
            print(f"  WARNING: Simulation took a long time ({sim_time:.1f} seconds)")
        else:
            print(f"  ✓ Simulation completed in reasonable time")
        
        # Display sample results
        print(f"\nSample results (first time point):")
        print(f"  Time: {result['t'][0]:.2f} s")
        print(f"  Distance: {result['dist'][0]:.2f} miles")
        
        print(f"  Coupler forces (N):")
        for i in range(min(5, F.shape[1])):
            print(f"    Group {i+1}: {F[0, i]:.0f}")
        if F.shape[1] > 5:
            print(f"    ... and {F.shape[1]-5} more groups")
        
        print(f"  Relative displacements (m):")
        for i in range(min(5, DX.shape[1])):
            print(f"    Group {i+1}: {DX[0, i]:.6f}")
        if DX.shape[1] > 5:
            print(f"    ... and {DX.shape[1]-5} more groups")
        
        if nan_found or inf_found:
            print("\n" + "="*60)
            print("TEST COMPLETED WITH WARNINGS!")
            print("="*60)
            return False
        else:
            print("\n" + "="*60)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            return True
        
    except Exception as e:
        print(f"ERROR during runFastSimM execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_runFastSimM_test()
    sys.exit(0 if success else 1)