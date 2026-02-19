#!/usr/bin/env python3
"""
Test script for calcRopeForces.py function
This script loads test data and validates the calcRopeForces function output.
"""

import json
import numpy as np
import os
import sys
import csv

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'pysim'))

from pysim.calcRopeForces import calcRopeForces

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
    """Load profile data from JSON format"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON data from {filename}: {e}")
        return None

def validate_data_structures(train_data, locos_data, track_data, profile_data):
    """Validate that all required data structures are present and well-formed."""
    print("Validating data structures...")
    
    # Validate Train data
    required_train_fields = ['NumCars', 'LocoPosition', 'LoadWeight', 'Davis_a', 'Davis_b', 'Davis_c', 'Length']
    for field in required_train_fields:
        if field not in train_data:
            print(f"ERROR: Missing required field '{field}' in Train data")
            return False
    
    print(f"Train data: {train_data['NumCars']} cars, locomotives at positions {train_data['LocoPosition']}")
    print(f"Car weights: {len(train_data['LoadWeight'])} values, Davis coefficients: a={train_data['Davis_a']}, b={train_data['Davis_b']}, c={train_data['Davis_c']}")
    
    # Validate Locos data
    if not isinstance(locos_data, list) or len(locos_data) == 0:
        print("ERROR: Locos data should be a non-empty list")
        return False
    
    print(f"Locomotive data: {len(locos_data)} locomotives")
    for i, loco in enumerate(locos_data):
        if 'Weight' not in loco:
            print(f"ERROR: Locomotive {i} missing 'Weight' field")
            return False
        print(f"  Loco {i+1}: {loco.get('Model', 'Unknown')} - {loco['Weight']} tons")
    
    # Validate Track data structure
    if not isinstance(track_data, dict):
        print("ERROR: Track data should be a dictionary")
        return False
    
    print(f"Track data contains: {list(track_data.keys())}")
    
    # Validate profile data structure
    if profile_data is None:
        print("ERROR: Profile data is missing")
        return False
    
    # Validate profile data
    required_keys = ['CNotch', 'Speed', 'Dist']
    if not all(key in profile_data for key in required_keys):
        print(f"ERROR: Profile data missing required keys. Has: {list(profile_data.keys())}, needs: {required_keys}")
        return False
    
    # Check that all arrays have the same length
    lengths = [len(profile_data[key]) for key in required_keys]
    if not all(length == lengths[0] for length in lengths):
        print(f"ERROR: Profile data arrays have different lengths: {dict(zip(required_keys, lengths))}")
        return False
    
    print(f"Profile data: {lengths[0]} points with keys {required_keys}")
    
    return True

def run_calcRopeForces_test():
    """Main test function for calcRopeForces."""
    print("="*60)
    print("TESTING calcRopeForces.py")
    print("="*60)
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test data files
    print("\nLoading test data files...")
    
    # Load Train configuration
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
    
    # Load Track data
    track_file = os.path.join(test_dir, 'TrackData.txt')
    track_data = load_json_data(track_file)
    if track_data is None:
        print("Failed to load Track data")
        return False
    
    # Load Profile data (CSV format)
    profile_file = os.path.join(test_dir, 'profile.txt')
    profile_data = load_json_data(profile_file)
    if profile_data is None:
        print("Failed to load Profile data")
        return False
    
    # Validate data structures
    if not validate_data_structures(train_data, locos_data, track_data, profile_data):
        print("Data validation failed")
        return False
    
    print("\n" + "="*60)
    print("RUNNING calcRopeForces FUNCTION")
    print("="*60)
    
    try:       
        # Call calcRopeForces function
        print("\nCalling calcRopeForces...")
        result = calcRopeForces(profile_data, track_data, train_data, locos_data, slew=False)
        
        # Unpack results
        F, Fuel, Flocos_only, Fgravloco, Fgrav, Fdrag, Finertia, Floco = result
        
        print("calcRopeForces completed successfully!")
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        print(f"Rope forces matrix shape: {F.shape}")
        print(f"Total fuel consumption: {Fuel[-1]:.2f} gallons")
        print(f"Locomotive forces shape: {Flocos_only.shape}")
        
        # Analyze force statistics
        print(f"\nForce statistics (lbf):")
        print(f"  Total rope forces range: {np.min(F):.0f} to {np.max(F):.0f}")
        print(f"  Locomotive forces range: {np.min(Flocos_only):.0f} to {np.max(Flocos_only):.0f}")
        print(f"  Gravity forces range: {np.min(Fgrav):.0f} to {np.max(Fgrav):.0f}")
        print(f"  Drag forces range: {np.min(Fdrag):.0f} to {np.max(Fdrag):.0f}")
        print(f"  Inertia forces range: {np.min(Finertia):.0f} to {np.max(Finertia):.0f}")
        
        # Check for reasonable values
        print(f"\nSanity checks:")
        max_force = np.max(np.abs(F))
        if max_force > 1e6:
            print(f"  WARNING: Very large forces detected ({max_force:.0f} lbf)")
        else:
            print(f"  ✓ Maximum force magnitude reasonable ({max_force:.0f} lbf)")
        
        if np.any(np.isnan(F)):
            print("  ERROR: NaN values detected in results")
            return False
        else:
            print("  ✓ No NaN values in results")
        
        if np.any(np.isinf(F)):
            print("  ERROR: Infinite values detected in results")
            return False
        else:
            print("  ✓ No infinite values in results")
        
        # Display sample results
        print(f"\nSample rope forces at first location (lbf):")
        sample_forces = F[0, :]
        for i, force in enumerate(sample_forces[:min(10, len(sample_forces))]):
            print(f"  Coupler {i+1}: {force:.0f}")
        if len(sample_forces) > 10:
            print(f"  ... and {len(sample_forces)-10} more couplers")
        
        print(f"\nSample locomotive forces (lbf):")
        for i in range(min(len(locos_data), Flocos_only.shape[1])):
            print(f"  Loco {i+1}: {Flocos_only[0, i]:.0f}")
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"ERROR during calcRopeForces execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_calcRopeForces_test()
    sys.exit(0 if success else 1)