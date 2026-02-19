#!/usr/bin/env python3
"""
test_db_storage.py - Test script for database storage functionality

This script tests the database storage functionality without running
full simulations, useful for debugging the storage code.
"""

import json
import numpy as np
from datetime import datetime, timezone
from .runPostRun import (
    get_database_connection, 
    store_simulation_result, 
    store_failed_simulation,
    clear_existing_results,
    calculate_performance_metrics,
    prepare_array_for_storage
)

def create_mock_simulation_output():
    """Create mock simulation output data for testing."""
    # Create some sample time-series data
    time_points = 100
    t = np.linspace(0, 1, time_points)  # 1 hour simulation
    dist = np.linspace(0, 60, time_points)  # 60 miles
    
    # Mock data for 2 car groups
    v = [
        np.random.normal(45, 5, time_points),  # Group 1 speeds around 45 mph
        np.random.normal(44, 5, time_points)   # Group 2 speeds around 44 mph
    ]
    
    # Mock coupler forces
    Frope = [
        np.random.normal(50000, 10000, time_points),  # Group 1 forces
        np.random.normal(45000, 8000, time_points)    # Group 2 forces
    ]
    
    # Mock accelerations
    a = [
        np.random.normal(0, 0.5, time_points),  # Group 1 acceleration
        np.random.normal(0, 0.5, time_points)   # Group 2 acceleration
    ]
    
    return {
        't': t,
        'dist': dist,
        'v': v,
        'Frope': Frope,
        'a': a,
        'dtsim': 45.5,  # Simulation time in seconds
        'cargroup': [1, 2]  # Car groups
    }

def create_mock_section_metadata():
    """Create mock section metadata for testing."""
    return {
        'section_id': 1,
        'start_idx': 100,
        'end_idx': 200,
        'first_timestamp': datetime.now(timezone.utc),
        'sub_trip_id': 12345,
        'duration_seconds': 3600.0,
        'distance_miles': 60.0,
        'avg_speed_mph': 45.0,
        'max_speed_mph': 55.0,
        'train_cars': 100,
        'train_length': 5500.0,
        'loco_positions': [1, 50, 100]
    }

def test_prepare_array_for_storage():
    """Test the array preparation function."""
    print("=== Testing prepare_array_for_storage ===")
    
    # Test numpy array
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = prepare_array_for_storage(np_array, 'speed', 'Test speed data')
    print(f"NumPy array result: {result}")
    
    # Test list
    list_data = [10, 20, 30, 40, 50]
    result = prepare_array_for_storage(list_data, 'time', 'Test time data')
    print(f"List result: {result}")
    
    # Test empty data
    result = prepare_array_for_storage(None, 'empty', 'Empty data')
    print(f"Empty data result: {result}")

def test_performance_metrics():
    """Test the performance metrics calculation."""
    print("\n=== Testing calculate_performance_metrics ===")
    
    mock_output = create_mock_simulation_output()
    metrics = calculate_performance_metrics(mock_output)
    
    print("Calculated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

def test_database_storage():
    """Test storing and retrieving simulation results."""
    print("\n=== Testing Database Storage ===")
    
    # Get database connection
    engine, session = get_database_connection()
    
    try:
        # Create test data
        tripsum_id = 999999  # Use a test trip ID that won't conflict
        mock_output = create_mock_simulation_output()
        mock_metadata = create_mock_section_metadata()
        sim_params = {
            'dcar': 5,
            'Ts': 0.5,
            'dx': 0.1,
            'X0': 'steady-state'
        }
        
        # Clear any existing test data
        print(f"Clearing existing test data for trip {tripsum_id}...")
        clear_existing_results(session, tripsum_id)
        
        # Test successful simulation storage
        print("Storing successful simulation result...")
        result_id = store_simulation_result(session, tripsum_id, mock_metadata, mock_output, sim_params)
        print(f"Stored simulation result with ID: {result_id}")
        
        # Test failed simulation storage
        print("Storing failed simulation result...")
        failed_metadata = mock_metadata.copy()
        failed_metadata['section_id'] = 2
        failed_id = store_failed_simulation(session, tripsum_id, failed_metadata, "Test error message", sim_params)
        print(f"Stored failed simulation with ID: {failed_id}")
        
        # Query back the results to verify
        from query_simulation_results import list_simulation_results, get_simulation_arrays
        
        print("\nQuerying stored results...")
        results_df = list_simulation_results(session, tripsum_id=tripsum_id)
        print("Stored results:")
        print(results_df[['id', 'section_id', 'status', 'distance_miles', 'max_coupler_force_lbf']].to_string(index=False))
        
        # Get array data for the successful simulation
        if result_id:
            print(f"\nQuerying arrays for result ID {result_id}...")
            arrays = get_simulation_arrays(session, result_id)
            print("Available arrays:")
            for array_type, array_info in arrays.items():
                print(f"  {array_type}: {array_info['length']} values ({array_info['units']})")
        
        # Clean up test data
        print(f"\nCleaning up test data for trip {tripsum_id}...")
        clear_existing_results(session, tripsum_id)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.close()

def main():
    """Run all tests."""
    print("Running database storage tests...\n")
    
    test_prepare_array_for_storage()
    test_performance_metrics()
    test_database_storage()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()