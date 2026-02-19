#!/usr/bin/env python3
"""
Test script for runSim.py - Complete simulation pipeline test
"""

import sys
import os
import json
import numpy as np
import time

# Add the parent directory to the path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runSim import runSim

def load_test_data():
    """Load all the test data files"""
    print("Loading test data files...")

    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Profile data
    profile_file = os.path.join(test_dir, 'profile.txt')
    with open(profile_file, 'r') as f:
        profile_data = json.load(f)
    
    # Spec data
    spec_file = os.path.join(test_dir, 'spec.txt')
    with open(spec_file, 'r') as f:
        spec_data = json.load(f)
    
    # Track data
    track_file = os.path.join(test_dir, 'TrackData.txt')
    with open(track_file, 'r') as f:
        track_data = json.load(f)
    
    # Train data
    train_file = os.path.join(test_dir, 'Train.txt')
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    # Locomotive data
    locos_file = os.path.join(test_dir, 'Locos.txt')
    with open(locos_file, 'r') as f:
        locos_data = json.load(f)
    
    return profile_data, spec_data, track_data, train_data, locos_data

def validate_complete_results(fs):
    """Validate the complete runSim results"""
    print("\n" + "="*60)
    print("COMPLETE SIMULATION RESULTS ANALYSIS")
    print("="*60)
    
    print(f"Output structure contains {len(fs)} fields:")
    for key, value in fs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list length {len(value)}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Check for the key fields that should be added by the pipeline
    required_fields = ['Fsi_allcars', 'LVratio', 'spec']
    missing_fields = []
    for field in required_fields:
        if field not in fs:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    else:
        print("✓ All required pipeline fields present")
    
    # Analyze the new fields
    print(f"\nComplete pipeline results:")
    print(f"  Individual car forces (Fsi_allcars): {fs['Fsi_allcars'].shape}")
    print(f"  L/V ratios: {fs['LVratio'].shape}")
    print(f"  Force range (all cars): {fs['Fsi_allcars'].min():.0f} to {fs['Fsi_allcars'].max():.0f} N")
    print(f"  L/V ratio range: {fs['LVratio'].min():.4f} to {fs['LVratio'].max():.4f}")
    
    # Check for reasonable values
    max_force = np.abs(fs['Fsi_allcars']).max()
    max_lv = np.abs(fs['LVratio']).max()
    
    if max_force > 5e6:  # 5 MN seems too high
        print(f"⚠️  WARNING: Very high forces detected ({max_force:.0f} N)")
    else:
        print(f"✓ Force magnitudes reasonable (max: {max_force:.0f} N)")
    
    if max_lv > 2.0:  # L/V ratios above 2.0 are extremely high
        print(f"⚠️  WARNING: Very high L/V ratios detected ({max_lv:.3f})")
    else:
        print(f"✓ L/V ratios reasonable (max: {max_lv:.3f})")
    
    # Check data integrity
    has_nan = np.any(np.isnan(fs['Fsi_allcars'])) or np.any(np.isnan(fs['LVratio']))
    has_inf = np.any(np.isinf(fs['Fsi_allcars'])) or np.any(np.isinf(fs['LVratio']))
    
    if has_nan:
        print("❌ NaN values detected in results")
        return False
    if has_inf:
        print("❌ Infinite values detected in results")
        return False
    
    print("✓ No NaN or infinite values detected")
    
    return True

def run_complete_simulation_test():
    """Run the complete simulation pipeline test"""
    print("="*60)
    print("TESTING COMPLETE SIMULATION PIPELINE (runSim)")
    print("="*60)
    
    # Load test data
    profile_data, spec_data, track_data, train_data, locos_data = load_test_data()
    
    print(f"Profile: {len(profile_data['Dist'])} points, {profile_data['Dist'][0]:.1f} to {profile_data['Dist'][-1]:.1f} miles")
    print(f"Train: {len(train_data['LocoPosition'])} locomotives, {train_data['NumCars']} cars")
    
    # Test parameters
    test_params = {
        'dcar': 5,
        'Ts': 0.5,
        'dx': 0.1,
        'range_': [98, 259],
        'X0': 'steady-state'
    }
    
    print(f"\nTest parameters:")
    print(f"  dcar (cars per group): {test_params['dcar']}")
    print(f"  Ts (time step): {test_params['Ts']}")
    print(f"  dx (distance step): {test_params['dx']}")
    print(f"  range: {test_params['range_']}")
    print(f"  X0 (initial state): {test_params['X0']}")
    
    # Run the complete simulation
    print(f"\n" + "="*40)
    print("RUNNING COMPLETE SIMULATION PIPELINE")
    print("="*40)
    
    start_time = time.time()
    try:
        fs = runSim(profile_data, spec_data, track_data, train_data, locos_data, **test_params)
        execution_time = time.time() - start_time
        
        print(f"\nComplete pipeline finished in {execution_time:.2f} seconds!")
        
        # Validate results
        if validate_complete_results(fs):
            print(f"\n✓ COMPLETE PIPELINE TEST PASSED!")
            return True
        else:
            print(f"\n❌ COMPLETE PIPELINE TEST FAILED!")
            return False
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ ERROR during complete pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_simulation_test()
    sys.exit(0 if success else 1)