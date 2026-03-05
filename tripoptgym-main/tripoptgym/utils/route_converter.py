"""
Route data conversion utilities.

Converts parsed route data from text format to structured CSV format with:
- Distance points at regular intervals
- Interpolated effective grades
- Step-wise speed limits
- Calculated elevations
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def parse_terrain_entity_table(lines: List[str]) -> pd.DataFrame:
    """
    Parse the Terrain Entity Table from route input.
    
    Args:
        lines: List of text lines from route input file
        
    Returns:
        DataFrame with columns: DIR, Sup_Elev, Grade, Curve
    """
    data = []
    in_table = False
    
    for line in lines:
        if 'Terrain Entity Table' in line:
            in_table = True
            continue
        
        if in_table:
            # Skip header lines
            if 'DIR' in line and 'Sup Elev' in line:
                continue
            if '---' in line:
                continue
            
            # Check if we've reached the next section
            if line.strip() and not line.strip().startswith('-') and 'Table' in line:
                break
            
            # Parse data line
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    dir_val = float(parts[0].strip())
                    sup_elev = float(parts[1].strip())
                    grade = float(parts[2].strip())
                    curve = float(parts[3].strip())
                    data.append({
                        'DIR': dir_val,
                        'Sup_Elev': sup_elev,
                        'Grade': grade,
                        'Curve': curve
                    })
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data)


def parse_effective_grade_table(lines: List[str]) -> pd.DataFrame:
    """
    Parse the EFFECTIVE_GRADE_TABLE from route input.
    
    Args:
        lines: List of text lines from route input file
        
    Returns:
        DataFrame with columns: Distance, Effective_Grade
    """
    data = []
    in_table = False
    
    for line in lines:
        if 'EFFECTIVE_GRADE_TABLE' in line:
            in_table = True
            continue
        
        if in_table:
            # Skip header lines
            if 'Distance In Route' in line:
                continue
            if '---' in line:
                continue
            
            # Check if we've reached the next section
            if line.strip() and not line.strip().startswith('-') and 'Table' in line:
                break
            
            # Parse data line
            parts = line.split('|')
            if len(parts) >= 2:
                try:
                    distance = float(parts[0].strip())
                    grade = float(parts[1].strip())
                    data.append({
                        'Distance': distance,
                        'Effective_Grade': grade
                    })
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data)


def parse_speed_limit_table(lines: List[str]) -> pd.DataFrame:
    """
    Parse the Speed Limit Entity Table from route input.
    
    Args:
        lines: List of text lines from route input file
        
    Returns:
        DataFrame with columns: DIR, Eff_Speed_Limit
    """
    data = []
    in_table = False
    
    for line in lines:
        if 'Speed Limit Entity Table' in line:
            in_table = True
            continue
        
        if in_table:
            # Skip header lines
            if 'DIR' in line and 'Civil SpdLim' in line:
                continue
            if '---' in line:
                continue
            
            # End of file or next section
            if not line.strip():
                break
            
            # Parse data line
            parts = line.split('|')
            if len(parts) >= 6:
                try:
                    dir_val = float(parts[0].strip())
                    eff_speed_lim = float(parts[5].strip())
                    data.append({
                        'DIR': dir_val,
                        'Eff_Speed_Limit': eff_speed_lim
                    })
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data)


def interpolate_values(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """
    Interpolate values using piecewise linear interpolation.
    
    Args:
        x_new: New x-coordinates for interpolation
        x_old: Original x-coordinates
        y_old: Original y-values
        
    Returns:
        Interpolated y-values at x_new positions
    """
    return np.interp(x_new, x_old, y_old)


def stepwise_lookup(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """
    Step-backward lookup for speed limits (stepwise constant interpolation).
    
    For each query point in x_new, finds the last x_old value that is <= query point
    and returns the corresponding y_old value. This maintains constant values between
    data points, which is appropriate for speed limits that change at discrete locations.
    
    Args:
        x_new: New x-coordinates for lookup
        x_old: Original x-coordinates (must be sorted)
        y_old: Original y-values
        
    Returns:
        Looked-up y-values at x_new positions
    """
    result = np.zeros(len(x_new))
    
    for i, x in enumerate(x_new):
        # Find the index where x_old > x (first point that exceeds current location)
        idx = np.searchsorted(x_old, x, side='right')
        
        if idx > 0:
            # Step back to get the speed limit at the previous point
            result[i] = y_old[idx - 1]
        else:
            # If we're before the first data point, use the first value
            result[i] = y_old[0]
    
    return result


def calculate_elevation(distances: np.ndarray, effective_grades: np.ndarray, terrain_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate elevation at each distance point by integrating grade.
    
    The elevation change is calculated as:
    - delta_elevation = (grade/100) * delta_distance * 5280
    - where grade is in percent, distance is in miles, and result is in feet
    
    Args:
        distances: Array of distance points (miles)
        effective_grades: Array of grades at each distance point (percent)
        terrain_df: Terrain dataframe with initial Sup_Elev value
        
    Returns:
        Array of elevations (feet)
    """
    elevations = np.zeros(len(distances))
    
    # Starting elevation from terrain table (Sup_Elev at first point)
    if len(terrain_df) > 0:
        elevations[0] = terrain_df.iloc[0]['Sup_Elev']
    
    # Calculate elevation changes by integrating grade
    for i in range(1, len(distances)):
        distance_interval = distances[i] - distances[i-1]  # in miles
        avg_grade = (effective_grades[i] + effective_grades[i-1]) / 2  # average grade in interval
        
        # Convert: distance in miles * 5280 feet/mile * grade/100 = elevation change in feet
        elevation_change = distance_interval * 5280 * (avg_grade / 100)
        elevations[i] = elevations[i-1] + elevation_change
    
    return elevations


def convert_route_data(input_file: str, output_file: str, step_size: float = 0.05) -> pd.DataFrame:
    """
    Convert parsed route data to structured CSV format.
    
    Args:
        input_file: Path to parsed route input text file
        output_file: Path to output CSV file
        step_size: Distance increment for output points (miles)
        
    Returns:
        DataFrame with converted route data
    """
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse all tables
    print("Parsing Terrain Entity Table...")
    terrain_df = parse_terrain_entity_table(lines)
    print(f"  Found {len(terrain_df)} terrain entries")
    
    print("Parsing Effective Grade Table...")
    grade_df = parse_effective_grade_table(lines)
    print(f"  Found {len(grade_df)} grade entries")
    
    print("Parsing Speed Limit Table...")
    speed_df = parse_speed_limit_table(lines)
    print(f"  Found {len(speed_df)} speed limit entries")
    
    # Determine distance range from terrain table
    min_dir = terrain_df['DIR'].min()
    max_dir = terrain_df['DIR'].max()
    print(f"\nDistance range: {min_dir:.4f} to {max_dir:.4f} miles")
    
    # Generate distance array with specified step size
    distances = np.arange(min_dir, max_dir + step_size, step_size)
    print(f"Generated {len(distances)} distance points")
    
    # Interpolate effective grades
    print("\nInterpolating effective grades...")
    effective_grades = interpolate_values(
        distances,
        grade_df['Distance'].values,
        grade_df['Effective_Grade'].values
    )
    
    # Lookup speed limits (stepwise constant, not interpolated)
    print("Looking up speed limits...")
    speed_limits = stepwise_lookup(
        distances,
        speed_df['DIR'].values,
        speed_df['Eff_Speed_Limit'].values
    )
    
    # Calculate elevations
    print("Calculating elevations...")
    elevations = calculate_elevation(distances, effective_grades, terrain_df)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'Distance In Route': distances,
        'Effective Grade Percent': effective_grades,
        'Effective Speed Limit': speed_limits,
        'Elevation': elevations
    })
    
    # Round to match expected precision
    output_df['Distance In Route'] = output_df['Distance In Route'].round(2)
    output_df['Effective Grade Percent'] = output_df['Effective Grade Percent'].round(4)
    output_df['Effective Speed Limit'] = output_df['Effective Speed Limit'].round(1)
    output_df['Elevation'] = output_df['Elevation'].round(2)
    
    # Save to CSV
    print(f"\nWriting to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print(f"Done! Created {len(output_df)} rows")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(output_df.head(10))
    
    return output_df
