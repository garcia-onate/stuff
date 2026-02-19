#!/usr/bin/env python3
"""
runPostRun.py - Post-run analysis script for train simulation

This script retrieves field data from a PostgreSQL database and runs FastSim
simulation using the recorded train data. It populates the profile and TrackData
structures from database records and calls runFastSimM() to perform the simulation.

The script queries the data_recorder table which contains 1-second samples of
train operational data including position, speed, notch commands, and track conditions.
"""

import json
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime
import os

from .runFastSimM import runFastSimM
from .models import DataRecorder, TripSummary, SubTrip
from .unit_conversions import lbf_

def load_json_data(filename):
    """Load and parse JSON data from file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def get_database_connection():
    """Create database connection and session."""
    engine = create_engine('postgresql://to-esudm:%25%5DL35eL%5BnRwk%40wbd@w-w1-at1-use1-dev-genai-rds.c3gsu026a7kf.us-east-1.rds.amazonaws.com:5432/to-esudm')
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session

def list_available_trips(session, limit=10):
    """
    List available trips with data recorder records.
    
    Args:
        session: SQLAlchemy session
        limit: Maximum number of trips to return
        
    Returns:
        list: Trip information with record counts
    """
    # Query for trips that have data recorder records
    query = text("""
        SELECT 
            ts.tripsum_id,
            ts.train_id,
            ts.lead_loco,
            ts.start_time_gmt,
            ts.total_trip_miles,
            COUNT(dr.id) as record_count
        FROM trip_summary ts
        INNER JOIN data_recorder dr ON ts.tripsum_id = dr.tripsum_id
        GROUP BY ts.tripsum_id, ts.train_id, ts.lead_loco, ts.start_time_gmt, ts.total_trip_miles
        HAVING COUNT(dr.id) > 0
        ORDER BY ts.start_time_gmt DESC
        LIMIT :limit
    """)
    
    result = session.execute(query, {"limit": limit})
    trips = []
    
    for row in result:
        trips.append({
            'tripsum_id': row.tripsum_id,
            'train_id': row.train_id,
            'lead_loco': row.lead_loco,
            'start_time_gmt': row.start_time_gmt,
            'total_trip_miles': row.total_trip_miles,
            'record_count': row.record_count
        })
    
    return trips

def fetch_trip_data(session, tripsum_id):
    """
    Fetch data recorder records for a specific trip.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID to query
        
    Returns:
        pandas.DataFrame: Data recorder records sorted by timestamp
    """
    query = session.query(DataRecorder).filter(
        DataRecorder.tripsum_id == tripsum_id
    ).order_by(DataRecorder.timestamp)
    
    # Convert to DataFrame for easier manipulation
    records = []
    for record in query:
        records.append({
            'timestamp': record.timestamp,
            'distance_miles': record.distance_miles or 0.0,
            'col_8828': record.col_8828 or 0.0,  # Speed limit
            'col_8831': record.col_8831 or 0,    # Grade
            'col_8809': record.col_8809 or 0.0,  # toNotch
            'col_20538': record.col_20538 or 0,  # DBEffortStatus
            'col_20612': record.col_20612 or 0,  # DpRemoteEngCmdFdbk
            'col_20559': record.col_20559 or 0.0, # Speed
            'col_8844': getattr(record, 'col_8844', None),  # Davis A coefficient
            'col_8845': getattr(record, 'col_8845', None),  # Davis B coefficient
            'col_8846': getattr(record, 'col_8846', None)   # Davis C coefficient
        })
    
    return pd.DataFrame(records)

def fetch_trip_summary_data(session, tripsum_id):
    """
    Fetch trip summary and sub trip data for a specific trip.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID to query
        
    Returns:
        tuple: (trip_summary, sub_trip) records
    """
    # Get trip summary
    trip_summary = session.query(TripSummary).filter(
        TripSummary.tripsum_id == tripsum_id
    ).first()
    
    # Get sub trip data (use first sub trip record if multiple exist)
    sub_trip = session.query(SubTrip).filter(
        SubTrip.tripsum_id == tripsum_id
    ).first()
    
    return trip_summary, sub_trip

def fetch_sub_trip_for_timestamp(session, tripsum_id, timestamp):
    """
    Fetch the appropriate sub trip record for a specific timestamp.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID to query
        timestamp: Timestamp to find the appropriate sub trip for
        
    Returns:
        SubTrip: Sub trip record that contains the given timestamp
    """
    # Get all sub trips for this trip, ordered by start time
    sub_trips = session.query(SubTrip).filter(
        SubTrip.tripsum_id == tripsum_id
    ).order_by(SubTrip.start_time_gmt).all()
    
    if not sub_trips:
        return None
    
    # Find the sub trip that starts closest to but before the timestamp
    best_match = None
    for sub_trip in sub_trips:
        if sub_trip.start_time_gmt and sub_trip.start_time_gmt <= timestamp:
            # This sub trip starts before or at our timestamp
            if best_match is None or sub_trip.start_time_gmt > best_match.start_time_gmt:
                best_match = sub_trip
    
    # If no sub trip starts before the timestamp, use the first one
    if best_match is None and sub_trips:
        best_match = sub_trips[0]
    
    return best_match

def create_train_structure_from_db(trip_summary, sub_trip, section_df):
    """
    Create Train data structure from database trip summary and sub trip records.
    
    Args:
        trip_summary: TripSummary database record
        sub_trip: SubTrip database record
        section_df: DataFrame with section data for Davis coefficients
        
    Returns:
        dict: Train structure compatible with FastSim
        
    Raises:
        ValueError: If required data is missing or invalid
    """
    if sub_trip is None:
        raise ValueError("No sub_trip data available for this timestamp")
    
    loads = sub_trip.loads
    empties = sub_trip.empties
    num_cars = loads + empties
    
    # Require weight and length data - no defaults
    if not sub_trip.weight or sub_trip.weight <= 0:
        raise ValueError(f"Invalid weight: {sub_trip.weight}")
    
    if not sub_trip.length or sub_trip.length <= 0:
        raise ValueError(f"Invalid length: {sub_trip.length}")
    
    total_weight = sub_trip.weight  # In tons
    total_length = sub_trip.length  # In feet
    
    # Create default coupler types (0 = standard knuckle coupler)
    coupler_types = [0] * num_cars
    
    # Calculate average car weight (tonnage is trailing tonnage, excludes locomotives)
    avg_car_weight = total_weight / num_cars  # tons per car
    
    # Create arrays for car properties
    load_weights = [avg_car_weight] * num_cars  # Weight in tons (cars only)
    
    # Create preload pattern: loaded cars first, then empty cars
    preload = [100000] * loads + [0] * empties
    
    # Parse locomotive positions from loco_details - this will raise ValueError if invalid
    loco_positions = parse_locomotive_positions(sub_trip.loco_details, num_cars)
    
    # Calculate car lengths excluding locomotive lengths (73 feet each)
    locomotive_length = 73  # feet per locomotive
    total_locomotive_length = len(loco_positions) * locomotive_length
    car_length_available = total_length - total_locomotive_length
    
    if car_length_available <= 0:
        raise ValueError(f"Total length ({total_length} ft) too small for {len(loco_positions)} locomotives ({total_locomotive_length} ft)")
    
    # Recalculate average car length excluding locomotives
    avg_car_length = car_length_available / num_cars
    load_lengths = [avg_car_length] * num_cars  # Length in feet (cars only)
    
    # Get Davis coefficients from data recorder fields (col_8844, col_8845, col_8846)
    davis_a, davis_b, davis_c = get_davis_coefficients_from_data(section_df)
    
    train_structure = {
        'CouplerType': coupler_types,
        'Davis_a': davis_a,
        'Davis_b': davis_b,
        'Davis_c': davis_c,
        'Length': total_length,
        'LoadLength': load_lengths,
        'LoadWeight': load_weights,
        'LocoPosition': loco_positions,
        'NumCars': num_cars,
        'PreLoad': preload
    }
    
    return train_structure

def parse_train_config(total_cars, loco_positions):
    """
    Parse train configuration from locomotive positions.
    
    Args:
        total_cars: Total number of cars in the train (freight cars only, excluding locomotives)
        loco_positions: List of locomotive positions (positions within total train length including locomotives)
        
    Returns:
        tuple: (train_type, cars_after_last_loco, num_consist)
    """
    if not loco_positions:
        return "invalid", 0, 0  # Return 0 for num_consist

    loco_positions = sorted(loco_positions)
    
    # Calculate total train length (cars + locomotives)
    total_train_length = total_cars + len(loco_positions)
    
    # Validate that all locomotive positions are within the total train length
    if any(pos > total_train_length for pos in loco_positions):
        return "invalid", 0, 0

    consists = []
    current_consist = [loco_positions[0]]

    # Identify consists (consecutive locomotive positions)
    for i in range(1, len(loco_positions)):
        if loco_positions[i] == loco_positions[i - 1] + 1:
            current_consist.append(loco_positions[i])
        else:
            consists.append(current_consist)
            current_consist = [loco_positions[i]]
    consists.append(current_consist)

    # Calculate number of consists
    num_consist = len(consists)

    # Determine train type
    has_lead = any(1 in consist for consist in consists)
    remote_consists = [c for c in consists if 1 not in c]

    # Calculate cars after the last locomotive
    last_loco_position = loco_positions[-1]
    cars_after_last_loco = total_train_length - last_loco_position
    
    if cars_after_last_loco < 0:
        return "invalid", 0, num_consist

    if has_lead and not remote_consists:
        return "conventional", cars_after_last_loco, num_consist
    elif remote_consists:
        if cars_after_last_loco == 0:
            return "end-dp", 0, num_consist
        else:
            return "mid-dp", cars_after_last_loco, num_consist
    else:
        return "invalid", 0, num_consist

def extract_positions(detail):
    """
    Extract locomotive positions from loco_details string.
    
    Args:
        detail: String containing locomotive configuration details
        
    Returns:
        list: List of locomotive positions
        
    Raises:
        ValueError: If detail cannot be parsed
    """
    positions = []
    for part in detail.split(';'):
        part = part.strip()
        if not part:
            raise ValueError("Empty part encountered in LOCO_DETAILS")
        if '-' not in part:
            raise ValueError(f"Missing '-' in part: '{part}'")
        num_str = part.split('-')[0].strip()
        if not num_str.isdigit():
            raise ValueError(f"Invalid integer value: '{num_str}' in part: '{part}'")
        positions.append(int(num_str))
    return positions

def parse_locomotive_positions(loco_details, num_cars):
    """
    Parse locomotive positions from loco_details string.
    
    Args:
        loco_details: String containing locomotive configuration details
        num_cars: Total number of cars in the train (freight cars only, excluding locomotives)
        
    Returns:
        list: List of locomotive positions
        
    Raises:
        ValueError: If loco_details cannot be parsed or results in invalid configuration
    """
    if not loco_details:
        raise ValueError("No locomotive details available")
    
    try:
        # Use the provided extract_positions function
        loco_positions = extract_positions(loco_details)
        
        # Validate locomotive positions are reasonable (should be positive)
        for pos in loco_positions:
            if pos < 1:
                raise ValueError(f"Locomotive position {pos} is invalid (must be >= 1)")
        
        # Validate train configuration using the provided logic
        train_type, cars_after_last_loco, num_consist = parse_train_config(num_cars, loco_positions)
        
        if train_type == "invalid":
            raise ValueError(f"Invalid train configuration: total_cars={num_cars}, loco_positions={loco_positions}")
            
    except Exception as e:
        raise ValueError(f"Failed to parse locomotive details '{loco_details}': {e}")
    
    return loco_positions

def get_davis_coefficients_from_data(section_df):
    """
    Extract Davis coefficients from data recorder fields.
    
    Args:
        section_df: DataFrame containing data recorder records
        
    Returns:
        tuple: (davis_a, davis_b, davis_c) coefficients
    """
    # Default values in case data is not available
    default_davis_a = 1.32
    default_davis_b = 0.011
    default_davis_c = 0.0006
    
    try:
        # Check if the Davis coefficient columns exist in the DataFrame
        if 'col_8844' in section_df.columns:
            davis_a_values = section_df['col_8844'].dropna()
            davis_a = davis_a_values.iloc[0] if len(davis_a_values) > 0 else default_davis_a
        else:
            davis_a = default_davis_a
            
        if 'col_8845' in section_df.columns:
            davis_b_values = section_df['col_8845'].dropna()
            davis_b = davis_b_values.iloc[0] if len(davis_b_values) > 0 else default_davis_b
        else:
            davis_b = default_davis_b
            
        if 'col_8846' in section_df.columns:
            davis_c_values = section_df['col_8846'].dropna()
            davis_c = davis_c_values.iloc[0] if len(davis_c_values) > 0 else default_davis_c
        else:
            davis_c = default_davis_c
            
    except Exception as e:
        print(f"Warning: Could not extract Davis coefficients from data: {e}")
        davis_a, davis_b, davis_c = default_davis_a, default_davis_b, default_davis_c
    
    return davis_a, davis_b, davis_c

def save_trip_data_to_csv(df, tripsum_id):
    """
    Save DataFrame to CSV file for debugging purposes.
    
    Args:
        df: DataFrame to save
        tripsum_id: Trip ID for filename
    """
    filename = f"trip_data_{tripsum_id}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_trip_data_from_csv(filename):
    """
    Load trip data from CSV file.
    
    Args:
        filename: CSV filename to load
        
    Returns:
        pandas.DataFrame: Data recorder records
    """
    try:
        df = pd.read_csv(filename, parse_dates=['timestamp'])
        print(f"Loaded {len(df)} records from {filename}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def calculate_distance_traveled(df):
    """
    Calculate cumulative distance traveled from distance_miles field.
    
    Args:
        df: DataFrame with distance_miles column
        
    Returns:
        numpy.array: Cumulative distance traveled
    """
    # Handle missing values
    distance_miles = df['distance_miles'].fillna(0)
    
    # Calculate cumulative distance
    calc_dist_traveled = distance_miles.cumsum().values
    
    return calc_dist_traveled

def calculate_notch_values(df):
    """
    Calculate lead notch and remote notch values from database fields.
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        tuple: (lead_notch, remote_notch) arrays
    """
    lead_notch = []
    remote_notch = []
    
    for _, row in df.iterrows():
        col_8809_val = row['col_8809'] if pd.notna(row['col_8809']) else 0
        col_20538_val = row['col_20538'] if pd.notna(row['col_20538']) else 0
        col_20612_val = row['col_20612'] if pd.notna(row['col_20612']) else 0
        
        # Calculate LeadNotch: toNotch + (DBEffortStatus / -12.5)
        lead_notch_val = int(col_8809_val) + int(col_20538_val / -12.5)
        lead_notch.append(lead_notch_val)
        
        # Calculate RemoteNotch based on DpRemoteEngCmdFdbk
        if int(col_20612_val) > 8:
            remote_notch_val = -1 * max(0, int(col_20612_val) - 11)
        else:
            remote_notch_val = int(col_20612_val)
        remote_notch.append(remote_notch_val)
    
    return np.array(lead_notch), np.array(remote_notch)

def create_track_data_structure(df, calc_dist_traveled):
    """
    Create TrackData structure from database records.
    
    Args:
        df: DataFrame with track data
        calc_dist_traveled: Cumulative distance array
        
    Returns:
        dict: TrackData structure
    """
    TrackData = {
        'Grade': {
            'Dist': calc_dist_traveled,
            'Percent': df['col_8831'].fillna(0).values
        },
        'SpdLim': {
            'Dist': calc_dist_traveled,
            'Mph': df['col_8828'].fillna(0).values
        },
        'PathDist': calc_dist_traveled[-1] if len(calc_dist_traveled) > 0 else 0.0
    }
    
    return TrackData

def create_profile_structure(df, calc_dist_traveled, lead_notch, remote_notch):
    """
    Create profile structure from database records.
    
    Args:
        df: DataFrame with profile data
        calc_dist_traveled: Cumulative distance array
        lead_notch: Lead notch values
        remote_notch: Remote notch values
        
    Returns:
        dict: Profile structure
    """
    # Calculate time in hours from timestamp
    if len(df) > 0:
        start_time = df['timestamp'].iloc[0]
        time_hours = [(ts - start_time).total_seconds() / 3600.0 for ts in df['timestamp']]
    else:
        time_hours = []
    
    # Create fence_flg array (placeholder - needs to be defined based on requirements)
    fence_flg = np.ones(len(df), dtype=int)  # Default to 1, adjust as needed
    
    profile = {
        'Dist': calc_dist_traveled,
        'leadNotch': lead_notch,
        'fence_flg': fence_flg,
        'remoteNotch': remote_notch,
        'Time': np.array(time_hours),
        'Speed': df['col_20559'].fillna(0).values
    }
    
    return profile

def find_contiguous_sections(df, speed_threshold=15.0, min_distance_miles=2.0):
    """
    Find contiguous sections where train speed > threshold.
    
    Args:
        df: DataFrame with speed data
        speed_threshold: Minimum speed threshold (default: 15 mph)
        min_distance_miles: Minimum distance for a valid section (default: 2.0 miles)
        
    Returns:
        list: List of tuples (start_idx, end_idx) for each contiguous section
    """
    if df.empty:
        return []
    
    # Create boolean mask for speed > threshold
    speed_mask = df['col_20559'].fillna(0) > speed_threshold
    
    # Find contiguous sections
    sections = []
    start_idx = None
    
    for i, is_above_threshold in enumerate(speed_mask):
        if is_above_threshold and start_idx is None:
            # Start of a new section
            start_idx = i
        elif not is_above_threshold and start_idx is not None:
            # End of current section
            end_idx = i - 1
            
            # Check if section meets minimum distance requirement
            if end_idx > start_idx:
                section_df = df.iloc[start_idx:end_idx+1]
                section_distance = calculate_distance_traveled(section_df)
                total_distance = section_distance[-1] - section_distance[0] if len(section_distance) > 0 else 0.0
                if total_distance >= min_distance_miles:
                    sections.append((start_idx, end_idx))
            
            start_idx = None
    
    # Handle case where section extends to end of data
    if start_idx is not None:
        end_idx = len(df) - 1
        section_df = df.iloc[start_idx:end_idx+1]
        section_distance = calculate_distance_traveled(section_df)
        total_distance = section_distance[-1] - section_distance[0] if len(section_distance) > 0 else 0.0
        if total_distance >= min_distance_miles:
            sections.append((start_idx, end_idx))
    
    return sections

def create_simulation_inputs(df, sections, session, tripsum_id, trip_summary):
    """
    Create simulation input data for each contiguous section.
    
    Args:
        df: Complete DataFrame
        sections: List of (start_idx, end_idx) tuples
        session: SQLAlchemy session for database queries
        tripsum_id: Trip summary ID for sub trip lookup
        trip_summary: Trip summary record
        
    Returns:
        list: List of simulation input dictionaries
    """
    simulation_inputs = []
    
    for i, (start_idx, end_idx) in enumerate(sections):
        # Extract section data
        section_df = df.iloc[start_idx:end_idx+1].copy().reset_index(drop=True)
        
        # Get timestamp of first record in this section
        first_timestamp = section_df.iloc[0]['timestamp']
        
        # Find the appropriate sub trip for this timestamp
        sub_trip = fetch_sub_trip_for_timestamp(session, tripsum_id, first_timestamp)
        
        # Create Train structure specific to this section
        try:
            train_structure = create_train_structure_from_db(trip_summary, sub_trip, section_df)
        except ValueError as e:
            # Mark this section as failed due to train structure issues
            simulation_input = {
                'section_id': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'failed': True,
                'failure_reason': str(e),
                'first_timestamp': first_timestamp,
                'sub_trip_id': sub_trip.id if sub_trip else None
            }
            simulation_inputs.append(simulation_input)
            continue
        
        # Calculate distance traveled for this section (starts from 0 for simulation)
        calc_dist_traveled = calculate_distance_traveled(section_df)
        
        # Get original trip distances for this section (for database storage)
        # Calculate cumulative distance from the beginning of the entire trip
        full_trip_distances = calculate_distance_traveled(df)
        section_start_distance = full_trip_distances[start_idx] if start_idx < len(full_trip_distances) else 0.0
        section_end_distance = full_trip_distances[end_idx] if end_idx < len(full_trip_distances) else 0.0
        
        # Calculate notch values for this section
        lead_notch, remote_notch = calculate_notch_values(section_df)
        
        # Create TrackData structure for this section
        track_data = create_track_data_structure(section_df, calc_dist_traveled)
        
        # Create profile structure for this section
        profile = create_profile_structure(section_df, calc_dist_traveled, lead_notch, remote_notch)
        
        # Create range for this section
        range_ = [calc_dist_traveled[0], calc_dist_traveled[-1]] if len(calc_dist_traveled) > 0 else [0.0, 0.0]
        
        simulation_input = {
            'section_id': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'profile': profile,
            'track_data': track_data,
            'train_structure': train_structure,
            'range': range_,
            'duration_seconds': (section_df.iloc[-1]['timestamp'] - section_df.iloc[0]['timestamp']).total_seconds(),
            'distance_miles': calc_dist_traveled[-1] - calc_dist_traveled[0] if len(calc_dist_traveled) > 0 else 0.0,
            'avg_speed_mph': section_df['col_20559'].mean(),
            'max_speed_mph': section_df['col_20559'].max(),
            'first_timestamp': first_timestamp,
            'sub_trip_id': sub_trip.id if sub_trip else None,
            # Original trip distance references for database storage
            'original_start_distance': section_start_distance,
            'original_end_distance': section_end_distance
        }
        
        simulation_inputs.append(simulation_input)
    
    return simulation_inputs

def create_locos_structure_for_train(base_locos, num_locomotives):
    """
    Create a Locos structure that matches the number of locomotives in the train.
    
    Args:
        base_locos: Original Locos structure from JSON file
        num_locomotives: Number of locomotives actually in the train
        
    Returns:
        list: Locos structure with correct number of locomotives
    """
    if not base_locos or len(base_locos) == 0:
        raise ValueError("Base Locos structure is empty")
    
    # Use the first locomotive model as the template
    template_loco = base_locos[0].copy()
    
    # Create new Locos structure with the required number of locomotives
    new_locos = []
    for i in range(num_locomotives):
        loco_copy = template_loco.copy()
        new_locos.append(loco_copy)
    
    return new_locos

def calculate_performance_metrics(simulation_output):
    """
    Calculate performance metrics from simulation output for database storage.
    
    Args:
        simulation_output: Output dictionary from runFastSimM
        
    Returns:
        dict: Dictionary of calculated performance metrics
    """
    metrics = {}
    
    try:
        # Coupler force metrics - using Fropesi (rope forces) and Fmaxsi/Fminsi
        if 'Fmaxsi' in simulation_output and 'Fminsi' in simulation_output:
            # Use the pre-calculated max/min values from FastSim
            fmax = simulation_output['Fmaxsi']
            fmin = simulation_output['Fminsi']
            if hasattr(fmax, '__len__') and len(fmax) > 0:
                metrics['max_coupler_force_lbf'] = float(np.max(fmax))
            else:
                metrics['max_coupler_force_lbf'] = float(fmax) if fmax is not None else None
                
            if hasattr(fmin, '__len__') and len(fmin) > 0:
                metrics['min_coupler_force_lbf'] = float(np.min(fmin))
            else:
                metrics['min_coupler_force_lbf'] = float(fmin) if fmin is not None else None
        elif 'Fropesi' in simulation_output and len(simulation_output['Fropesi']) > 0:
            # Fallback to Fropesi arrays - structured as [time_steps][couplers]
            fropesi_data = simulation_output['Fropesi']
            all_forces = []
            
            # Flatten all force values across all time steps and couplers
            for time_step in fropesi_data:
                if hasattr(time_step, '__len__'):
                    all_forces.extend(time_step)
                else:
                    all_forces.append(time_step)
            
            if all_forces:
                metrics['max_coupler_force_lbf'] = float(np.max(all_forces))
                metrics['min_coupler_force_lbf'] = float(np.min(all_forces))
            else:
                metrics['max_coupler_force_lbf'] = None
                metrics['min_coupler_force_lbf'] = None
        else:
            metrics['max_coupler_force_lbf'] = None
            metrics['min_coupler_force_lbf'] = None
        
        # Speed metrics - using DVsi (velocity arrays)
        # DVsi is structured as [time_steps][car_groups]
        if 'DVsi' in simulation_output and len(simulation_output['DVsi']) > 0:
            dvsi_data = simulation_output['DVsi']
            all_speeds = []
            
            # Flatten all speed values across all time steps and car groups
            for time_step in dvsi_data:
                if hasattr(time_step, '__len__'):
                    all_speeds.extend(time_step)
                else:
                    all_speeds.append(time_step)
            
            if all_speeds:
                metrics['max_speed_achieved_mph'] = float(np.max(all_speeds))
            else:
                metrics['max_speed_achieved_mph'] = None
        else:
            metrics['max_speed_achieved_mph'] = None
        
        # Energy calculation (placeholder - would need power data)
        # This would require integrating power over time
        metrics['total_energy_mwh'] = None
        
    except Exception as e:
        print(f"Warning: Error calculating performance metrics: {e}")
        metrics = {
            'max_coupler_force_lbf': None,
            'min_coupler_force_lbf': None,
            'max_speed_achieved_mph': None,
            'total_energy_mwh': None
        }
    
    return metrics

def prepare_array_for_storage(array_data, array_type, description=None):
    """
    Prepare numpy array or list for JSONB storage.
    
    Args:
        array_data: Numpy array or list to store
        array_type: Type identifier for the array
        description: Optional description
        
    Returns:
        dict: Dictionary ready for JSONB storage
    """
    if array_data is None:
        return None
    
    # Convert numpy array to list if needed
    if isinstance(array_data, np.ndarray):
        values = array_data.tolist()
        data_type = str(array_data.dtype)
    else:
        values = list(array_data) if hasattr(array_data, '__iter__') else [array_data]
        data_type = 'list'
    
    # Handle NaN and infinity values (PostgreSQL JSON doesn't support them)
    cleaned_values = []
    for value in values:
        if isinstance(value, float):
            if np.isnan(value):
                cleaned_values.append(None)  # Convert NaN to null
            elif np.isinf(value):
                cleaned_values.append(None)  # Convert infinity to null
            else:
                cleaned_values.append(value)
        else:
            cleaned_values.append(value)
    
    # Determine units based on array type
    units_map = {
        'time': 'hours',
        'distance': 'miles', 
        'speed': 'mph',
        'coupler_forces': 'lbf',
        'limited': 'lbf',  # for limited_coupler_forces arrays
        'rope': 'lbf',     # for rope_forces arrays
        'acceleration': 'mph/s',
        'power': 'hp',
        'elevation': 'feet',
        'grade': 'percent',
        'position': 'feet'
    }
    
    # Extract base type for units mapping
    base_type = array_type.split('_')[0] if '_' in array_type else array_type
    
    return {
        'values': cleaned_values,
        'units': units_map.get(base_type, 'unknown'),
        'length': len(cleaned_values),
        'data_type': data_type,
        'description': description or f'{array_type} data'
    }

def store_simulation_result(session, tripsum_id, section_metadata, simulation_output, sim_params):
    """
    Store simulation result and arrays in the database.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID
        section_metadata: Metadata from simulation input
        simulation_output: Output from runFastSimM
        sim_params: Simulation parameters (dcar, Ts, dx, X0)
        
    Returns:
        int: ID of the stored simulation result record
    """
    # Optional debug output (uncomment for debugging)
    # print(f"    DEBUG: Simulation output keys: {list(simulation_output.keys())}")
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(simulation_output)
    
    # Helper function to convert numpy types to native Python types
    def convert_to_native_type(value):
        if value is None:
            return None
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    # Prepare simulation results record
    result_data = {
        'tripsum_id': tripsum_id,
        'section_id': section_metadata['section_id'],
        'start_idx': section_metadata['start_idx'],
        'end_idx': section_metadata['end_idx'],
        'first_timestamp': section_metadata['first_timestamp'],
        'sub_trip_id': section_metadata['sub_trip_id'],
        'duration_seconds': convert_to_native_type(section_metadata['duration_seconds']),
        'distance_miles': convert_to_native_type(section_metadata['distance_miles']),
        'avg_speed_mph': convert_to_native_type(section_metadata['avg_speed_mph']),
        'max_speed_mph': convert_to_native_type(section_metadata['max_speed_mph']),
        'train_cars': section_metadata.get('train_structure', {}).get('NumCars', None),
        'train_length': convert_to_native_type(section_metadata.get('train_structure', {}).get('Length', None)),
        'num_locomotives': len(section_metadata.get('loco_positions', [])),
        'dcar': sim_params['dcar'],
        'ts_sample': convert_to_native_type(sim_params['Ts']),
        'dx_step': convert_to_native_type(sim_params['dx']),
        'x0_initial': str(sim_params['X0']),
        'simulation_time_seconds': convert_to_native_type(simulation_output.get('dtsim', 0.0)),
        'total_time_steps': len(simulation_output.get('t', [])),
        'start_distance_miles': convert_to_native_type(section_metadata.get('original_start_distance', 0.0)),
        'end_distance_miles': convert_to_native_type(section_metadata.get('original_end_distance', 0.0)),
        'num_car_groups': len(simulation_output.get('cargroup', [])),
        'status': 'completed'
    }
    
    # Add performance metrics (convert numpy types)
    converted_metrics = {k: convert_to_native_type(v) for k, v in metrics.items()}
    result_data.update(converted_metrics)
    
    # Insert simulation result record
    insert_query = text("""
        INSERT INTO postrun_results (
            tripsum_id, section_id, start_idx, end_idx, first_timestamp, sub_trip_id,
            duration_seconds, distance_miles, avg_speed_mph, max_speed_mph,
            train_cars, train_length, num_locomotives,
            dcar, ts_sample, dx_step, x0_initial,
            simulation_time_seconds, total_time_steps, start_distance_miles, end_distance_miles, num_car_groups,
            max_coupler_force_lbf, min_coupler_force_lbf, max_speed_achieved_mph, total_energy_mwh,
            status
        ) VALUES (
            :tripsum_id, :section_id, :start_idx, :end_idx, :first_timestamp, :sub_trip_id,
            :duration_seconds, :distance_miles, :avg_speed_mph, :max_speed_mph,
            :train_cars, :train_length, :num_locomotives,
            :dcar, :ts_sample, :dx_step, :x0_initial,
            :simulation_time_seconds, :total_time_steps, :start_distance_miles, :end_distance_miles, :num_car_groups,
            :max_coupler_force_lbf, :min_coupler_force_lbf, :max_speed_achieved_mph, :total_energy_mwh,
            :status
        ) RETURNING id
    """)
    
    result = session.execute(insert_query, result_data)
    simulation_result_id = result.fetchone()[0]
    
    # Store array data
    arrays_to_store = [
        ('time', simulation_output.get('t', []), 'Time vector in hours'),
        ('distance', simulation_output.get('dist', []), 'Distance vector in miles')
    ]
    
    # Store limited coupler force arrays (Fsi - limited coupler forces for each car group)
    # Fsi is structured as [time_steps][couplers], need to transpose to [couplers][time_steps]
    # Convert from Newtons to lbf
    if 'Fsi' in simulation_output:
        fsi_data = simulation_output['Fsi']
        if len(fsi_data) > 0 and hasattr(fsi_data[0], '__len__'):
            num_couplers = len(fsi_data[0])
            
            # Transpose the data: convert [time_steps][couplers] to [couplers][time_steps]
            for coupler_idx in range(num_couplers):
                force_time_series_n = [time_step[coupler_idx] for time_step in fsi_data if len(time_step) > coupler_idx]
                # Convert from Newtons to lbf
                force_time_series_lbf = [force_n / lbf_() for force_n in force_time_series_n]
                arrays_to_store.append((
                    f'limited_coupler_forces_{coupler_idx+1}', 
                    force_time_series_lbf, 
                    f'Limited coupler forces for coupler {coupler_idx+1} in lbf'
                ))
    
    # Store rope force arrays (Fropesi - rope forces for each car group)
    # Fropesi is structured as [time_steps][couplers], need to transpose to [couplers][time_steps]
    # Convert from Newtons to lbf
    if 'Fropesi' in simulation_output:
        fropesi_data = simulation_output['Fropesi']
        if len(fropesi_data) > 0 and hasattr(fropesi_data[0], '__len__'):
            num_couplers = len(fropesi_data[0])
            
            # Transpose the data: convert [time_steps][couplers] to [couplers][time_steps]
            for coupler_idx in range(num_couplers):
                force_time_series_n = [time_step[coupler_idx] for time_step in fropesi_data if len(time_step) > coupler_idx]
                # Convert from Newtons to lbf
                force_time_series_lbf = [force_n / lbf_() for force_n in force_time_series_n]
                arrays_to_store.append((
                    f'rope_forces_{coupler_idx+1}', 
                    force_time_series_lbf, 
                    f'Rope forces for coupler {coupler_idx+1} in lbf'
                ))
    
    # Insert array records
    for array_type, array_data, description in arrays_to_store:
        if array_data is not None and len(array_data) > 0:
            prepared_data = prepare_array_for_storage(array_data, array_type, description)
            if prepared_data:
                array_insert_query = text("""
                    INSERT INTO postrun_arrays (
                        postrun_result_id, array_type, array_data, array_length, data_type, description
                    ) VALUES (
                        :postrun_result_id, :array_type, :array_data, :array_length, :data_type, :description
                    )
                """)
                
                session.execute(array_insert_query, {
                    'postrun_result_id': simulation_result_id,
                    'array_type': array_type,
                    'array_data': json.dumps(prepared_data),
                    'array_length': prepared_data['length'],
                    'data_type': prepared_data['data_type'],
                    'description': description
                })
    
    session.commit()
    return simulation_result_id

def store_failed_simulation(session, tripsum_id, section_metadata, error_message, sim_params):
    """
    Store a record for a failed simulation.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID
        section_metadata: Metadata from simulation input
        error_message: Error message describing the failure
        sim_params: Simulation parameters (dcar, Ts, dx, X0)
        
    Returns:
        int: ID of the stored simulation result record
    """
    result_data = {
        'tripsum_id': tripsum_id,
        'section_id': section_metadata['section_id'],
        'start_idx': section_metadata.get('start_idx'),
        'end_idx': section_metadata.get('end_idx'),
        'first_timestamp': section_metadata.get('first_timestamp'),
        'sub_trip_id': section_metadata.get('sub_trip_id'),
        'duration_seconds': section_metadata.get('duration_seconds'),
        'distance_miles': section_metadata.get('distance_miles'),
        'avg_speed_mph': section_metadata.get('avg_speed_mph'),
        'max_speed_mph': section_metadata.get('max_speed_mph'),
        'train_cars': section_metadata.get('train_structure', {}).get('NumCars', None),
        'train_length': section_metadata.get('train_structure', {}).get('Length', None),
        'dcar': sim_params['dcar'],
        'ts_sample': sim_params['Ts'],
        'dx_step': sim_params['dx'],
        'x0_initial': str(sim_params['X0']),
        'status': 'failed',
        'error_message': error_message
    }
    
    insert_query = text("""
        INSERT INTO postrun_results (
            tripsum_id, section_id, start_idx, end_idx, first_timestamp, sub_trip_id,
            duration_seconds, distance_miles, avg_speed_mph, max_speed_mph,
            train_cars, train_length,
            dcar, ts_sample, dx_step, x0_initial,
            status, error_message
        ) VALUES (
            :tripsum_id, :section_id, :start_idx, :end_idx, :first_timestamp, :sub_trip_id,
            :duration_seconds, :distance_miles, :avg_speed_mph, :max_speed_mph,
            :train_cars, :train_length,
            :dcar, :ts_sample, :dx_step, :x0_initial,
            :status, :error_message
        ) RETURNING id
    """)
    
    result = session.execute(insert_query, result_data)
    simulation_result_id = result.fetchone()[0]
    session.commit()
    return simulation_result_id

def clear_existing_results(session, tripsum_id):
    """
    Clear existing simulation results for a trip before running new simulations.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID to clear results for
    """
    # Delete arrays first due to foreign key constraint
    session.execute(
        text("DELETE FROM postrun_arrays WHERE postrun_result_id IN (SELECT id FROM postrun_results WHERE tripsum_id = :tripsum_id)"),
        {'tripsum_id': tripsum_id}
    )
    
    # Delete simulation results
    session.execute(
        text("DELETE FROM postrun_results WHERE tripsum_id = :tripsum_id"),
        {'tripsum_id': tripsum_id}
    )
    
    session.commit()
    print(f"Cleared existing simulation results for trip {tripsum_id}")

def run_post_run_analysis(tripsum_id, dcar=None, Ts=None, dx=None, range_=None, X0=None, csv_file=None, speed_threshold=15.0, min_distance_miles=2.0, store_in_db=True, save_csv=True, clear_existing=False):
    """
    Main function to run post-run analysis for a specific trip.
    Runs multiple simulations for contiguous sections where speed > threshold.
    
    Args:
        tripsum_id: Trip summary ID to analyze
        dcar: Number of cars grouped for FastSim (optional)
        Ts: Sample time for outputs (optional) 
        dx: Distance step size in miles (optional)
        range_: Distance range to simulate [start, end] (optional)
        X0: Initial states (optional)
        csv_file: CSV file to load data from instead of database (optional)
        speed_threshold: Minimum speed threshold for sections (default: 15 mph)
        min_distance_miles: Minimum distance for valid sections (default: 2.0 miles)
        store_in_db: Whether to store results in database (default: True)
        save_csv: Whether to save trip data to CSV file (default: True)
        clear_existing: Whether to clear existing results before storing new ones (default: False)
        
    Returns:
        dict: Dictionary containing simulation results and metadata
    """
    print(f"Starting post-run analysis for trip {tripsum_id}")

    if dcar is None:
        dcar = 5

    if Ts is None:
        Ts = 0.5

    if dx is None:
        dx = 0.1

    if X0 is None:
        X0 = 'steady-state'
        
    # Get database connection (used for both data and Train structure)
    engine, session = get_database_connection()
    
    try:
        # Load data from CSV or database
        if csv_file:
            print(f"Loading data from CSV file: {csv_file}")
            df = load_trip_data_from_csv(csv_file)
            if df is None:
                print(f"Failed to load data from {csv_file}")
                return {
                    'tripsum_id': tripsum_id,
                    'sections_found': 0,
                    'simulation_results': [],
                    'total_records': len(df)
                }
            print(f"Loaded {len(df)} data records from CSV")
        else:
            # Fetch trip data from database
            print("Fetching data from database...")
            df = fetch_trip_data(session, tripsum_id)
            
            if df.empty:
                print(f"No data found for tripsum_id {tripsum_id}")
                return {
                    'tripsum_id': tripsum_id,
                    'sections_found': 0,
                    'simulation_results': [],
                    'total_records': len(df)
                }
            
            print(f"Retrieved {len(df)} data records")
            
            # Save to CSV for future debugging if enabled
            if save_csv:
                save_trip_data_to_csv(df, tripsum_id)


        # Get the absolute path to the directory this script is in
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the full path to the data files (now in package data directory)
        locos_path = os.path.join(script_dir, 'data', 'Locos.txt')
        spec_path = os.path.join(script_dir, 'data', 'spec.txt')

        # Load static structures (base Locos and spec)
        print("Loading base Locos structure...")
        base_Locos = load_json_data(locos_path)
        if base_Locos is None:
            print("Failed to load Locos.txt")
            return {
                'tripsum_id': tripsum_id,
                'sections_found': 0,
                'simulation_results': [],
                'total_records': len(df)
            }
        
        print("Loading spec structure...")
        spec = load_json_data(spec_path)
        if spec is None:
            print("Failed to load spec.txt")
            return {
                'tripsum_id': tripsum_id,
                'sections_found': 0,
                'simulation_results': [],
                'total_records': len(df)
            }

        # Get trip summary for Train structure creation
        print("Fetching trip summary data...")
        trip_summary, _ = fetch_trip_summary_data(session, tripsum_id)
        
        # Find contiguous sections where speed > threshold
        print(f"Finding contiguous sections where speed > {speed_threshold} mph...")
        sections = find_contiguous_sections(df, speed_threshold, min_distance_miles)
        
        if not sections:
            print(f"No contiguous sections found with speed > {speed_threshold} mph and distance > {min_distance_miles} miles")
            return {
                'tripsum_id': tripsum_id,
                'sections_found': 0,
                'simulation_results': [],
                'total_records': len(df)
            }
        
        print(f"Found {len(sections)} contiguous sections")
        
        # Create simulation inputs for each section (with section-specific Train structures)
        print("Creating simulation inputs for each section...")
        simulation_inputs = create_simulation_inputs(df, sections, session, tripsum_id, trip_summary)
        
        # Clear existing results if requested
        if store_in_db and clear_existing:
            clear_existing_results(session, tripsum_id)
        
    finally:
        # Keep session open for storing results
        pass
    
    # Prepare simulation parameters for storage
    sim_params = {
        'dcar': dcar,
        'Ts': Ts,
        'dx': dx,
        'X0': X0
    }
    
    # Run simulations for each section
    simulation_results = []
    stored_result_ids = []
    
    for i, sim_input in enumerate(simulation_inputs):
        print(f"\n--- Running simulation {i+1}/{len(simulation_inputs)} ---")
        
        # Check if this section failed during input creation
        if sim_input.get('failed', False):
            print(f"Section {sim_input['section_id']}: FAILED - {sim_input['failure_reason']}")
            print(f"  First timestamp: {sim_input['first_timestamp']}")
            print(f"  Sub trip ID: {sim_input['sub_trip_id']}")
            
            # Store failed simulation record if database storage is enabled
            if store_in_db:
                try:
                    failed_metadata = {
                        'section_id': sim_input['section_id'],
                        'start_idx': sim_input['start_idx'],
                        'end_idx': sim_input['end_idx'],
                        'first_timestamp': sim_input['first_timestamp'],
                        'sub_trip_id': sim_input['sub_trip_id']
                    }
                    failed_id = store_failed_simulation(session, tripsum_id, failed_metadata, sim_input['failure_reason'], sim_params)
                    stored_result_ids.append(failed_id)
                    print(f"  Stored failure record with ID: {failed_id}")
                except Exception as e:
                    print(f"  Warning: Failed to store failure record: {e}")
            
            continue
        
        print(f"Section {sim_input['section_id']}: {sim_input['duration_seconds']:.1f}s, "
              f"{sim_input['distance_miles']:.2f} miles, avg speed: {sim_input['avg_speed_mph']:.1f} mph")
        print(f"  First timestamp: {sim_input['first_timestamp']}")
        print(f"  Sub trip ID: {sim_input['sub_trip_id']}")
        print(f"  Train: {sim_input['train_structure']['NumCars']} cars, {sim_input['train_structure']['Length']} ft")
        
        try:
            # Create section-specific Locos structure that matches the number of locomotives in this train
            num_locomotives = len(sim_input['train_structure']['LocoPosition'])
            section_Locos = create_locos_structure_for_train(base_Locos, num_locomotives)
            print(f"  Created Locos structure for {num_locomotives} locomotives")
            
            # Create a copy of spec and override the Train fields to match our actual train configuration
            section_spec = spec.copy()
            section_spec['Train'] = sim_input['train_structure'].copy()
            print(f"  Updated spec['Train'] to match actual train configuration")
            
            # Run FastSim simulation for this section using section-specific Train, Locos, and spec structures
            output = runFastSimM(
                sim_input['profile'], 
                section_spec, 
                sim_input['track_data'], 
                sim_input['train_structure'], 
                section_Locos, 
                dcar=dcar, 
                Ts=Ts, 
                dx=dx, 
                range_=sim_input['range'], 
                X0=X0
            )
            
            # Add metadata to the output
            output['section_metadata'] = {
                'section_id': sim_input['section_id'],
                'start_idx': sim_input['start_idx'],
                'end_idx': sim_input['end_idx'],
                'duration_seconds': sim_input['duration_seconds'],
                'distance_miles': sim_input['distance_miles'],
                'avg_speed_mph': sim_input['avg_speed_mph'],
                'max_speed_mph': sim_input['max_speed_mph'],
                'first_timestamp': sim_input['first_timestamp'],
                'sub_trip_id': sim_input['sub_trip_id'],
                'train_cars': sim_input['train_structure']['NumCars'],
                'train_length': sim_input['train_structure']['Length']
            }
            
            simulation_results.append(output)
            print(f"Section {sim_input['section_id']} simulation completed successfully")
            
            # Store simulation results in database if enabled
            if store_in_db:
                try:
                    # Add locomotive positions to metadata for storage
                    sim_input['loco_positions'] = sim_input['train_structure']['LocoPosition']
                    
                    result_id = store_simulation_result(session, tripsum_id, sim_input, output, sim_params)
                    stored_result_ids.append(result_id)
                    print(f"  Stored results in database with ID: {result_id}")
                except Exception as e:
                    print(f"  Warning: Failed to store simulation results: {e}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in section {sim_input['section_id']} simulation: {error_msg}")
            
            # Store failed simulation record if database storage is enabled
            if store_in_db:
                try:
                    failed_id = store_failed_simulation(session, tripsum_id, sim_input, error_msg, sim_params)
                    stored_result_ids.append(failed_id)
                    print(f"  Stored failure record with ID: {failed_id}")
                except Exception as store_e:
                    print(f"  Warning: Failed to store failure record: {store_e}")
            
            # Continue with next section even if one fails
            continue
    
    # Close database session
    session.close()
    
    # Count failed sections
    failed_sections = sum(1 for sim_input in simulation_inputs if sim_input.get('failed', False))
    
    print(f"\nCompleted {len(simulation_results)} simulations out of {len(simulation_inputs)} sections")
    if failed_sections > 0:
        print(f"Failed sections due to train structure issues: {failed_sections}")
    
    if store_in_db and stored_result_ids:
        print(f"Stored {len(stored_result_ids)} records in database with IDs: {stored_result_ids}")
    
    # Return combined results
    return {
        'tripsum_id': tripsum_id,
        'sections_found': len(sections),
        'successful_simulations': len(simulation_results),
        'failed_sections': failed_sections,
        'simulation_results': simulation_results,
        'stored_result_ids': stored_result_ids if store_in_db else [],
        'total_records': len(df),
        'speed_threshold': speed_threshold,
        'min_distance_miles': min_distance_miles,
        'database_storage_enabled': store_in_db
    }

def main():
    """
    Example usage of the post-run analysis script.
    """
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run post-run analysis for train simulation')
    parser.add_argument('tripsum_id', type=int, help='Trip summary ID to analyze')
    parser.add_argument('--csv', type=str, help='Load data from CSV file instead of database')
    parser.add_argument('--dcar', type=int, help='Number of cars grouped for FastSim')
    parser.add_argument('--ts', type=float, help='Sample time for outputs')
    parser.add_argument('--dx', type=float, help='Distance step size in miles')
    parser.add_argument('--speed-threshold', type=float, default=15.0, help='Minimum speed threshold for sections (default: 15 mph)')
    parser.add_argument('--min-distance', type=float, default=2.0, help='Minimum distance for valid sections in miles (default: 2.0)')
    parser.add_argument('--no-db-storage', action='store_true', help='Disable database storage of results')
    parser.add_argument('--no-csv-storage', action='store_true', help='Disable automatic CSV file saving of trip data')
    parser.add_argument('--clear-existing', action='store_true', help='Clear existing results for this trip before storing new ones')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nUsage examples:")
        print("  python runPostRun.py 535")
        print("  python runPostRun.py 535 --csv trip_data_535.csv")
        print("  python runPostRun.py 535 --csv trip_data_535.csv --dcar 10 --ts 1.0")
        print("  python runPostRun.py 535 --no-csv-storage --no-db-storage")
        return
    
    try:
        # Run the analysis
        print(f"\nRunning analysis for trip {args.tripsum_id}...")
        results = run_post_run_analysis(
            args.tripsum_id, 
            dcar=args.dcar, 
            Ts=args.ts, 
            dx=args.dx,
            csv_file=args.csv,
            speed_threshold=args.speed_threshold,
            min_distance_miles=args.min_distance,
            store_in_db=not args.no_db_storage,
            save_csv=not args.no_csv_storage,
            clear_existing=args.clear_existing
        )
        
        # Print summary results
        print("\n=== Analysis Summary ===")
        print(f"Trip ID: {results['tripsum_id']}")
        print(f"Total data records: {results['total_records']}")
        print(f"Speed threshold: {results['speed_threshold']} mph")
        print(f"Minimum distance: {results['min_distance_miles']} miles")
        print(f"Contiguous sections found: {results['sections_found']}")
        print(f"Successful simulations: {results['successful_simulations']}")
        print(f"Database storage: {'Enabled' if results['database_storage_enabled'] else 'Disabled'}")
        if results['database_storage_enabled'] and results['stored_result_ids']:
            print(f"Stored database record IDs: {results['stored_result_ids']}")
        
        # Print detailed results for each simulation
        if results['simulation_results']:
            print("\n=== Individual Simulation Results ===")
            for i, sim_result in enumerate(results['simulation_results']):
                metadata = sim_result['section_metadata']
                print(f"\nSection {metadata['section_id']}:")
                print(f"  Duration: {metadata['duration_seconds']:.1f} seconds")
                print(f"  Distance: {metadata['distance_miles']:.2f} miles")
                print(f"  Average speed: {metadata['avg_speed_mph']:.1f} mph")
                print(f"  Max speed: {metadata['max_speed_mph']:.1f} mph")
                print(f"  Simulation time: {sim_result['dtsim']:.2f} seconds")
                print(f"  Time steps: {len(sim_result['t'])}")
                print(f"  Distance range: {sim_result['dist'][0]:.2f} to {sim_result['dist'][-1]:.2f} miles")
                print(f"  Car groups: {len(sim_result['cargroup'])}")
        else:
            print("\nNo simulations were completed successfully.")
        
        # Additional analysis can be added here
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()