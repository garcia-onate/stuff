#!/usr/bin/env python3
"""
Utility script to process dr_input.csv and generate dr_output.csv

This script:
1. Reads dr_input.csv with 1-second sample data
2. Calculates high-resolution route data (distance, elevation, etc.)
3. Samples the data at 0.05 mile intervals
4. Writes the result to dr_output.csv
"""

import pandas as pd
import numpy as np


def process_dr_data(input_file='dr_input.csv', output_file='dr_output.csv'):
    """
    Process DR input data and generate output sampled at 0.05 mile intervals.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Read input data
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # Rename columns for clarity
    column_mapping = {
        '8828': 'effective_speed_limit_mph',
        '11947': 'effective_grade_percent',
        '8831': 'raw_grade_percent',
        '8949': 'system_state_enum',
        '20559': 'speed_mph'
    }
    df.rename(columns=column_mapping, inplace=True)

    print(f"Processing {len(df)} samples...")

    # Calculate distance traveled in each 1-second interval
    # Speed is in mph, convert to miles per second (divide by 3600)
    df['distance_delta'] = df['speed_mph'] / 3600.0

    # Calculate cumulative distance
    df['distance_in_route'] = df['distance_delta'].cumsum()

    # Calculate elevation change for each segment
    # rise/run where rise = (raw_grade % / 100) * run
    # and run is the distance traveled
    # Convert to feet by multiplying by 5280 (feet per mile)
    df['elevation_delta'] = (df['raw_grade_percent'] / 100.0) * df['distance_delta'] * 5280.0

    # Calculate cumulative elevation (in feet)
    df['elevation'] = df['elevation_delta'].cumsum()

    # Create high-resolution output dataframe
    high_res_df = pd.DataFrame({
        'Distance In Route': df['distance_in_route'],
        'Effective Grade Percent': df['effective_grade_percent'],
        'Effective Speed Limit': df['effective_speed_limit_mph'],
        'Elevation': df['elevation']
    })

    print(f"High-resolution data: {len(high_res_df)} records")
    print(f"Total distance: {high_res_df['Distance In Route'].iloc[-1]:.2f} miles")
    print(f"Elevation change: {high_res_df['Elevation'].iloc[-1]:.2f} (units)")

    # Sample at 0.05 mile intervals
    sample_interval = 0.05  # miles
    max_distance = high_res_df['Distance In Route'].iloc[-1]

    # Create target distances for sampling
    target_distances = np.arange(0, max_distance + sample_interval, sample_interval)

    # Interpolate data at target distances
    sampled_data = []
    for target_dist in target_distances:
        if target_dist > max_distance:
            break

        # Find the closest data point or interpolate
        if target_dist == 0:
            # First point
            sampled_data.append({
                'Distance In Route': 0,
                'Effective Grade Percent': high_res_df['Effective Grade Percent'].iloc[0],
                'Effective Speed Limit': high_res_df['Effective Speed Limit'].iloc[0],
                'Elevation': 0
            })
        else:
            # Find surrounding points for interpolation
            idx = high_res_df['Distance In Route'].searchsorted(target_dist)

            if idx >= len(high_res_df):
                idx = len(high_res_df) - 1

            if idx == 0:
                # Use first point if target is before first sample
                row = high_res_df.iloc[0]
                sampled_data.append({
                    'Distance In Route': target_dist,
                    'Effective Grade Percent': row['Effective Grade Percent'],
                    'Effective Speed Limit': row['Effective Speed Limit'],
                    'Elevation': row['Elevation']
                })
            else:
                # Interpolate between idx-1 and idx
                row_before = high_res_df.iloc[idx-1]
                row_after = high_res_df.iloc[idx]

                dist_before = row_before['Distance In Route']
                dist_after = row_after['Distance In Route']

                # Linear interpolation factor
                if dist_after > dist_before:
                    t = (target_dist - dist_before) / (dist_after - dist_before)
                else:
                    t = 0

                sampled_data.append({
                    'Distance In Route': target_dist,
                    'Effective Grade Percent': row_before['Effective Grade Percent'] * (1-t) + row_after['Effective Grade Percent'] * t,
                    'Effective Speed Limit': row_before['Effective Speed Limit'] * (1-t) + row_after['Effective Speed Limit'] * t,
                    'Elevation': row_before['Elevation'] * (1-t) + row_after['Elevation'] * t
                })

    # Create output dataframe
    output_df = pd.DataFrame(sampled_data)

    print(f"Sampled data: {len(output_df)} records at {sample_interval} mile intervals")

    # Write to CSV
    print(f"Writing to {output_file}...")
    output_df.to_csv(output_file, index=False)

    print("Done!")
    print(f"\nOutput summary:")
    print(f"  Records: {len(output_df)}")
    print(f"  Distance range: 0 to {output_df['Distance In Route'].iloc[-1]:.2f} miles")
    print(f"  Elevation range: {output_df['Elevation'].min():.2f} to {output_df['Elevation'].max():.2f}")


if __name__ == '__main__':
    import sys

    # Allow command line arguments for input/output files
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dr_input.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'dr_output.csv'

    process_dr_data(input_file, output_file)
