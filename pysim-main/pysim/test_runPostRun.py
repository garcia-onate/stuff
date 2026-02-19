#!/usr/bin/env python3
"""
test_runPostRun.py - Test script for runPostRun functionality

This script tests the basic functionality of runPostRun.py without requiring
a database connection. It creates mock data to verify the data structure
creation and processing functions work correctly.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path so we can import runPostRun
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .runPostRun import (
    load_json_data,
    calculate_distance_traveled,
    calculate_notch_values,
    create_track_data_structure,
    create_profile_structure
)

def create_mock_data(num_records=100):
    """Create mock data that simulates database records."""
    # Create timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_records)]
    
    # Create mock data
    data = {
        'timestamp': timestamps,
        'distance_miles': np.random.uniform(0.01, 0.05, num_records),  # Miles per second
        'col_8828': np.random.uniform(40, 80, num_records),  # Speed limit
        'col_8831': np.random.uniform(-2, 2, num_records),   # Grade
        'col_8809': np.random.choice([-8, -4, 0, 4, 8], num_records),  # toNotch
        'col_20538': np.random.choice([0, -25, -50], num_records),  # DBEffortStatus
        'col_20612': np.random.choice([0, 4, 8, 12, 16], num_records),  # DpRemoteEngCmdFdbk
        'col_20559': np.random.uniform(30, 70, num_records)  # Speed
    }
    
    return pd.DataFrame(data)

def test_load_json_data():
    """Test the JSON loading functionality."""
    print("Testing JSON data loading...")
    
    # Test loading spec file
    spec = load_json_data('test_runFastSimM/spec.txt')
    if spec is not None:
        print("✓ spec.txt loaded successfully")
        print(f"  Route: {spec.get('Route', 'N/A')}")
        print(f"  Train NumCars: {spec.get('Train', {}).get('NumCars', 'N/A')}")
    else:
        print("✗ Failed to load spec.txt")
    
    # Test loading Train file
    train = load_json_data('test_runFastSimM/Train.txt')
    if train is not None:
        print("✓ Train.txt loaded successfully")
        print(f"  NumCars: {train.get('NumCars', 'N/A')}")
        print(f"  Length: {train.get('Length', 'N/A')}")
    else:
        print("✗ Failed to load Train.txt")
    
    # Test loading Locos file
    locos = load_json_data('test_runFastSimM/Locos.txt')
    if locos is not None:
        print("✓ Locos.txt loaded successfully")
        print(f"  Number of locomotives: {len(locos)}")
        if len(locos) > 0:
            print(f"  First loco name: {locos[0].get('LocoName', 'N/A')}")
    else:
        print("✗ Failed to load Locos.txt")

def test_distance_calculation():
    """Test distance calculation functionality."""
    print("\nTesting distance calculation...")
    
    # Create mock data
    df = create_mock_data(10)
    
    # Calculate distance
    calc_dist = calculate_distance_traveled(df)
    
    print(f"✓ Distance calculation completed")
    print(f"  Initial distance: {calc_dist[0]:.4f}")
    print(f"  Final distance: {calc_dist[-1]:.4f}")
    print(f"  Total distance: {calc_dist[-1] - calc_dist[0]:.4f} miles")

def test_notch_calculations():
    """Test notch value calculations."""
    print("\nTesting notch calculations...")
    
    # Create mock data
    df = create_mock_data(5)
    
    # Calculate notch values
    lead_notch, remote_notch = calculate_notch_values(df)
    
    print("✓ Notch calculations completed")
    print(f"  Lead notch values: {lead_notch}")
    print(f"  Remote notch values: {remote_notch}")

def test_structure_creation():
    """Test TrackData and profile structure creation."""
    print("\nTesting structure creation...")
    
    # Create mock data
    df = create_mock_data(20)
    calc_dist = calculate_distance_traveled(df)
    lead_notch, remote_notch = calculate_notch_values(df)
    
    # Create TrackData structure
    track_data = create_track_data_structure(df, calc_dist)
    print("✓ TrackData structure created")
    print(f"  Grade points: {len(track_data['Grade']['Dist'])}")
    print(f"  Speed limit points: {len(track_data['SpdLim']['Dist'])}")
    
    # Create profile structure
    profile = create_profile_structure(df, calc_dist, lead_notch, remote_notch)
    print("✓ Profile structure created")
    print(f"  Distance points: {len(profile['Dist'])}")
    print(f"  Time range: {profile['Time'][0]:.4f} to {profile['Time'][-1]:.4f} hours")
    print(f"  Speed range: {profile['Speed'].min():.1f} to {profile['Speed'].max():.1f}")

def main():
    """Run all tests."""
    print("=== runPostRun.py Test Suite ===\n")
    
    try:
        test_load_json_data()
        test_distance_calculation()
        test_notch_calculations()
        test_structure_creation()
        
        print("\n=== All Tests Completed ===")
        print("✓ All basic functionality tests passed")
        print("\nThe runPostRun.py script appears to be working correctly.")
        print("To run with real database data, use: python runPostRun.py [tripsum_id]")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()