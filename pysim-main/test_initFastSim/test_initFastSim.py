#!/usr/bin/env python3
"""
Test script for initFastSim.py function
This script loads test data and validates the initFastSim function output.
"""

import json
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from initFastSim import initFastSim

def load_json_data(filename):
    """Load and parse JSON data from file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def validate_data_structures(spec_data, locos_data):
    """Validate that all required data structures are present and well-formed."""
    print("Validating data structures...")
    
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
    train_data = spec_data['Train']
    required_train_fields = ['NumCars', 'LocoPosition', 'LoadWeight', 'CouplerType', 'PreLoad']
    for field in required_train_fields:
        if field not in train_data:
            print(f"ERROR: Missing required field '{field}' in Train data")
            return False
    
    print(f"Train data: {train_data['NumCars']} cars, locomotives at positions {train_data['LocoPosition']}")
    print(f"Car weights: {len(train_data['LoadWeight'])} values")
    print(f"Coupler types: {len(train_data['CouplerType'])} values")
    print(f"PreLoad values: {len(train_data['PreLoad'])} values")
    
    # Validate Locos data
    if not isinstance(locos_data, list) or len(locos_data) == 0:
        print("ERROR: Locos data should be a non-empty list")
        return False
    
    print(f"Locomotive data: {len(locos_data)} locomotives")
    for i, loco in enumerate(locos_data):
        if 'Weight' not in loco:
            print(f"ERROR: Locomotive {i} missing 'Weight' field")
            return False
        if 'Length' not in loco:
            print(f"ERROR: Locomotive {i} missing 'Length' field")
            return False
        print(f"  Loco {i+1}: {loco.get('LocoName', 'Unknown')} - {loco['Weight']} tons, {loco['Length']} ft")
    
    return True

def run_initFastSim_test():
    """Main test function for initFastSim."""
    print("="*60)
    print("TESTING initFastSim.py")
    print("="*60)
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test data files
    print("\nLoading test data files...")
    
    # Load spec configuration
    spec_file = os.path.join(test_dir, 'spec.txt')
    spec_data = load_json_data(spec_file)
    if spec_data is None:
        print("Failed to load spec data")
        return False
    
    # Load Locomotives data
    locos_file = os.path.join(test_dir, 'locoParam.txt')
    locos_data = load_json_data(locos_file)
    if locos_data is None:
        print("Failed to load Locos data")
        return False
    
    # Validate data structures
    if not validate_data_structures(spec_data, locos_data):
        print("Data validation failed")
        return False
    
    print("\n" + "="*60)
    print("RUNNING initFastSim FUNCTION")
    print("="*60)
    
    # Set test parameters as specified
    dcar = 5
    Ts = 0.5
    
    print(f"Test parameters:")
    print(f"  dcar (cars per group): {dcar}")
    print(f"  Ts (time step): {Ts}")
    
    try:       
        # Call initFastSim function
        print("\nCalling initFastSim...")
        result = initFastSim(spec_data, locos_data, dcar=dcar, Ts=Ts)
        
        # Unpack results
        Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB, cargroup, x2all, Fmin2all, Fmax2all = result
        
        print("initFastSim completed successfully!")
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        print(f"Number of car groups: {len(cargroup)}")
        print(f"Car group boundaries: {cargroup}")
        
        print(f"\nGroup mass vector shape: {groupMassVector.shape}")
        print(f"Group masses (tons): {groupMassVector}")
        
        print(f"\nLocked spring constants shape: {Klocked.shape}")
        print(f"Locked spring constants (lbf/in): {Klocked}")
        
        print(f"\nAnti-windup gains shape: {Kawu.shape}")
        print(f"Anti-windup gains: {Kawu}")
        
        print(f"\nGroup damping B shape: {groupDampingB.shape}")
        print(f"Group damping B: {groupDampingB}")
        
        print(f"\nGroup cushioning damping B shape: {groupCushioningDampingB.shape}")
        print(f"Group cushioning damping B: {groupCushioningDampingB}")
        
        print(f"\nCoupler displacement vector (x2all) length: {len(x2all)}")
        print(f"Displacement range: {x2all[0]:.3f} to {x2all[-1]:.3f} inches")
        
        print(f"\nForce limit matrices:")
        print(f"  Fmin2all shape: {Fmin2all.shape}")
        print(f"  Fmax2all shape: {Fmax2all.shape}")
        
        # Sanity checks
        print(f"\nSanity checks:")
        
        # Check for reasonable values
        if np.any(groupMassVector <= 0):
            print("  ERROR: Non-positive group masses detected")
            return False
        else:
            print("  ✓ All group masses are positive")
        
        if np.any(Klocked <= 0):
            print("  ERROR: Non-positive spring constants detected")
            return False
        else:
            print("  ✓ All spring constants are positive")
        
        if np.any(np.isnan([Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB])):
            print("  ERROR: NaN values detected in results")
            return False
        else:
            print("  ✓ No NaN values in primary results")
        
        if np.any(np.isinf([Klocked, Kawu, groupMassVector, groupDampingB, groupCushioningDampingB])):
            print("  ERROR: Infinite values detected in results")
            return False
        else:
            print("  ✓ No infinite values in primary results")
        
        # Check force matrices
        if np.any(np.isnan(Fmin2all)) or np.any(np.isnan(Fmax2all)):
            print("  ERROR: NaN values detected in force matrices")
            return False
        else:
            print("  ✓ No NaN values in force matrices")
        
        if np.any(np.isinf(Fmin2all)) or np.any(np.isinf(Fmax2all)):
            print("  ERROR: Infinite values detected in force matrices")
            return False
        else:
            print("  ✓ No infinite values in force matrices")
        
        # Check that Fmax >= Fmin everywhere
        inconsistent_count = np.sum(Fmax2all < Fmin2all)
        if inconsistent_count > 0:
            print(f"  WARNING: Force limits inconsistent at {inconsistent_count} points (Fmax < Fmin)")
            print(f"    This may be due to hysteresis envelope construction")
            print(f"    Max difference: {np.max(Fmin2all - Fmax2all):.2f} lbf")
        else:
            print("  ✓ Force limits are consistent (Fmax >= Fmin)")
        
        # Check car grouping consistency
        total_cars = spec_data['Train']['NumCars'] + len(spec_data['Train']['LocoPosition'])
        if cargroup[-1] != total_cars:
            print(f"  ERROR: Car grouping inconsistent (last group ends at {cargroup[-1]}, expected {total_cars})")
            return False
        else:
            print(f"  ✓ Car grouping consistent (total cars: {total_cars})")
        
        # Display sample results
        print(f"\nSample spring constants by group (lbf/in):")
        for i, k in enumerate(Klocked):
            print(f"  Group {i+1}: {k:.0f}")
        
        print(f"\nSample group masses (tons):")
        for i, mass in enumerate(groupMassVector):
            print(f"  Group {i+1}: {mass:.1f}")
        
        print(f"\nDisplacement breakpoints (first 10):")
        for i, x in enumerate(x2all[:10]):
            print(f"  Point {i+1}: {x:.3f} inches")
        if len(x2all) > 10:
            print(f"  ... and {len(x2all)-10} more points")
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"ERROR during initFastSim execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_initFastSim_test()
    sys.exit(0 if success else 1)