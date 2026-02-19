#!/usr/bin/env python3
"""
Test script for gravityforces.py function.

This script loads test data from JSON and CSV files in the test_gravityforces directory
and tests the gravityforces function using the FastSim data structures.
"""

import numpy as np
import json
import os
import sys

# Add parent directory to path to import gravityforces
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravityforces import gravityforces


def load_json_data(filepath):
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {filepath}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_csv_data(filepath):
    """Load CSV data from file."""
    try:
        data = np.loadtxt(filepath, delimiter=',')
        print(f"Successfully loaded {filepath} - shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def print_data_summary(data, name):
    """Print summary information about data structure."""
    print(f"\n--- {name} Summary ---")
    if isinstance(data, dict):
        print(f"Type: Dictionary with {len(data)} keys:")
        for key in data.keys():
            if isinstance(data[key], (list, np.ndarray)):
                if hasattr(data[key], 'shape'):
                    print(f"  {key}: array shape {data[key].shape}")
                else:
                    print(f"  {key}: list length {len(data[key])}")
            else:
                print(f"  {key}: {type(data[key]).__name__}")
    elif isinstance(data, list):
        print(f"Type: List with {len(data)} elements")
        if len(data) > 0:
            print(f"  First element type: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                print(f"  First element keys: {list(data[0].keys())}")
    elif isinstance(data, np.ndarray):
        print(f"Type: NumPy array")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    else:
        print(f"Type: {type(data).__name__}")


def validate_data_structures(dataset, trainParam, locoParam, distout):
    """Validate that the data structures match expected FastSim format."""
    print("\n=== Data Structure Validation ===")
    
    # Validate dataset structure
    required_dataset_keys = ['Grade', 'SpdLim', 'GPS', 'Milepost', 'PathDist']
    if not all(key in dataset for key in required_dataset_keys):
        missing = [key for key in required_dataset_keys if key not in dataset]
        print(f"WARNING: Missing dataset keys: {missing}")
    else:
        print("✓ Dataset structure valid")
    
    # Validate Grade structure
    if 'Grade' in dataset:
        grade_keys = ['Dist', 'Percent']
        grade_missing = [key for key in grade_keys if key not in dataset['Grade']]
        if grade_missing:
            print(f"WARNING: Missing Grade keys: {grade_missing}")
        else:
            print("✓ Grade structure valid")
    
    # Validate trainParam structure
    required_train_keys = ['LoadWeight', 'LoadLength', 'LocoPosition']
    train_missing = [key for key in required_train_keys if key not in trainParam]
    if train_missing:
        print(f"WARNING: Missing trainParam keys: {train_missing}")
    else:
        print("✓ TrainParam structure valid")
    
    # Validate locoParam structure
    if isinstance(locoParam, list) and len(locoParam) > 0:
        required_loco_keys = ['Weight', 'Length']
        loco_missing = [key for key in required_loco_keys if key not in locoParam[0]]
        if loco_missing:
            print(f"WARNING: Missing locoParam keys: {loco_missing}")
        else:
            print("✓ LocoParam structure valid")
    
    # Check data consistency
    if 'LocoPosition' in trainParam and isinstance(locoParam, list):
        num_locos_expected = len(trainParam['LocoPosition'])
        num_locos_actual = len(locoParam)
        if num_locos_expected != num_locos_actual:
            print(f"WARNING: Locomotive count mismatch - expected {num_locos_expected}, got {num_locos_actual}")
        else:
            print("✓ Locomotive count consistent")
    
    print("Validation complete.\n")


def main():
    """Main test function."""
    print("=== Testing gravityforces.py ===\n")
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test data files
    print("Loading test data files...")
    
    # Load JSON files
    dataset = load_json_data(os.path.join(test_dir, 'dataset.txt'))
    trainParam = load_json_data(os.path.join(test_dir, 'trainParam.txt'))
    locoParam = load_json_data(os.path.join(test_dir, 'locoParam.txt'))
    
    # Load CSV file
    distout = load_csv_data(os.path.join(test_dir, 'distout.csv'))
    
    # Check if all data loaded successfully
    if dataset is None or trainParam is None or locoParam is None or distout is None:
        print("ERROR: Failed to load some test data files. Exiting.")
        return
    
    # Print data summaries
    print_data_summary(dataset, "Dataset")
    print_data_summary(trainParam, "TrainParam")
    print_data_summary(locoParam, "LocoParam")
    print_data_summary(distout, "Distout")
    
    # Validate data structures
    validate_data_structures(dataset, trainParam, locoParam, distout)
    
    # Convert lists to numpy arrays where needed
    print("Converting data types for compatibility...")
    
    # Ensure trainParam arrays are numpy arrays
    for key in ['LoadWeight', 'LoadLength', 'LocoPosition']:
        if key in trainParam and isinstance(trainParam[key], list):
            trainParam[key] = np.array(trainParam[key])
    
    # Ensure dataset Grade arrays are numpy arrays  
    if 'Grade' in dataset:
        for key in ['Dist', 'Percent']:
            if key in dataset['Grade'] and isinstance(dataset['Grade'][key], list):
                dataset['Grade'][key] = np.array(dataset['Grade'][key])
    
    # Add PathDist if missing (required by makeeffelev)
    if 'PathDist' not in dataset:
        if 'Grade' in dataset and 'Dist' in dataset['Grade']:
            dataset['PathDist'] = float(np.max(dataset['Grade']['Dist']))
            print(f"Added PathDist: {dataset['PathDist']}")
    
    print("Data conversion complete.\n")
    
    # Test the gravityforces function
    print("=== Testing gravityforces function ===")
    
    try:
        print(f"Calling gravityforces with:")
        print(f"  - Train cars: {len(trainParam['LoadWeight'])}")
        print(f"  - Locomotives: {len(locoParam)}")
        print(f"  - Distance points: {len(distout)}")
        print(f"  - Track distance range: [{np.min(distout):.1f}, {np.max(distout):.1f}] miles")
        
        # Call gravityforces function
        F = gravityforces(dataset, trainParam, locoParam, distout)
        
        print(f"\n✓ gravityforces completed successfully!")
        print(f"Output shape: {F.shape}")
        print(f"Expected shape: ({len(distout)}, {len(trainParam['LoadWeight']) + len(locoParam)})")
        
        # Analyze results
        print(f"\nForce analysis:")
        print(f"  Force range: [{np.min(F):.1f}, {np.max(F):.1f}] lbf")
        print(f"  Mean absolute force: {np.mean(np.abs(F)):.1f} lbf")
        print(f"  Standard deviation: {np.std(F):.1f} lbf")
        
        # Check for reasonable values
        max_reasonable_force = 500000  # 500k lbf should be reasonable upper bound
        if np.max(np.abs(F)) > max_reasonable_force:
            print(f"WARNING: Some forces exceed {max_reasonable_force} lbf")
        else:
            print("✓ Force magnitudes appear reasonable")
        
        # Save results for inspection
        output_file = os.path.join(test_dir, 'gravityforces_output.csv')
        np.savetxt(output_file, F, delimiter=',', fmt='%.6f',
                   header=f'Gravity forces output - shape: {F.shape}, distances: {len(distout)} points')
        print(f"Results saved to: {output_file}")
        
        # Print sample results
        print(f"\nSample results (first 5 distance points, first 5 units):")
        print("Distance (mi)  | Forces (lbf)")
        print("-" * 50)
        for i in range(min(5, len(distout))):
            forces_str = " ".join([f"{F[i,j]:8.1f}" for j in range(min(5, F.shape[1]))])
            print(f"{distout[i]:12.1f} | {forces_str}")
        
        print(f"\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"ERROR in gravityforces function: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()