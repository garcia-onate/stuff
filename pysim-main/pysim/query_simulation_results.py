#!/usr/bin/env python3
"""
query_simulation_results.py - Query and display stored simulation results

This script provides functions to query the simulation results database
and retrieve stored simulation data for analysis and visualization.
"""

import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def get_database_connection():
    """Create database connection and session."""
    engine = create_engine('postgresql://to-esudm:%25%5DL35eL%5BnRwk%40wbd@w-w1-at1-use1-dev-genai-rds.c3gsu026a7kf.us-east-1.rds.amazonaws.com:5432/to-esudm')
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session

def list_simulation_results(session, tripsum_id=None, limit=20):
    """
    List simulation results, optionally filtered by trip ID.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Optional trip ID to filter by
        limit: Maximum number of results to return
        
    Returns:
        pandas.DataFrame: Simulation results summary
    """
    where_clause = "WHERE tripsum_id = :tripsum_id" if tripsum_id else ""
    params = {"limit": limit}
    if tripsum_id:
        params["tripsum_id"] = tripsum_id
    
    query = text(f"""
        SELECT 
            id,
            tripsum_id,
            section_id,
            created_at,
            status,
            duration_seconds,
            distance_miles,
            avg_speed_mph,
            max_speed_mph,
            train_cars,
            train_length,
            num_locomotives,
            simulation_time_seconds,
            total_time_steps,
            start_distance_miles,
            end_distance_miles,
            max_coupler_force_lbf,
            min_coupler_force_lbf,
            max_speed_achieved_mph,
            error_message
        FROM postrun_results 
        {where_clause}
        ORDER BY tripsum_id, section_id
        LIMIT :limit
    """)
    
    result = session.execute(query, params)
    return pd.DataFrame(result.fetchall(), columns=result.keys())

def get_simulation_arrays(session, simulation_result_id, array_types=None):
    """
    Get array data for a specific simulation result.
    
    Args:
        session: SQLAlchemy session
        simulation_result_id: ID of the simulation result
        array_types: List of array types to retrieve (optional, gets all if None)
        
    Returns:
        dict: Dictionary of array type -> array data
    """
    where_clause = "AND array_type = ANY(:array_types)" if array_types else ""
    params = {"postrun_result_id": simulation_result_id}
    if array_types:
        params["array_types"] = array_types
    
    query = text(f"""
        SELECT array_type, array_data, description
        FROM postrun_arrays 
        WHERE postrun_result_id = :postrun_result_id {where_clause}
        ORDER BY array_type
    """)
    
    result = session.execute(query, params)
    arrays = {}
    
    for row in result:
        # PostgreSQL JSONB is already parsed as dict, no need to json.loads
        if isinstance(row.array_data, dict):
            array_data = row.array_data
        else:
            array_data = json.loads(row.array_data)
        
        arrays[row.array_type] = {
            'values': array_data['values'],
            'units': array_data['units'],
            'description': row.description,
            'length': len(array_data['values'])
        }
    
    return arrays

def get_trip_summary(session, tripsum_id):
    """
    Get summary of all simulation results for a trip.
    
    Args:
        session: SQLAlchemy session
        tripsum_id: Trip summary ID
        
    Returns:
        dict: Trip summary information
    """
    # Get basic trip info
    trip_query = text("""
        SELECT 
            COUNT(*) as total_sections,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sections,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sections,
            SUM(duration_seconds) as total_duration_seconds,
            SUM(distance_miles) as total_distance_miles,
            AVG(avg_speed_mph) as overall_avg_speed_mph,
            MAX(max_speed_mph) as overall_max_speed_mph,
            MAX(max_coupler_force_lbf) as max_coupler_force_lbf,
            MIN(min_coupler_force_lbf) as min_coupler_force_lbf,
            MAX(created_at) as last_updated
        FROM postrun_results 
        WHERE tripsum_id = :tripsum_id
    """)
    
    result = session.execute(trip_query, {"tripsum_id": tripsum_id})
    row = result.fetchone()
    if row:
        summary = dict(row._mapping)
        summary['tripsum_id'] = tripsum_id
    else:
        summary = {'tripsum_id': tripsum_id, 'error': 'No results found'}
    
    return summary

def export_arrays_to_csv(session, simulation_result_id, output_prefix):
    """
    Export array data to CSV files for external analysis.
    
    Args:
        session: SQLAlchemy session
        simulation_result_id: ID of the simulation result
        output_prefix: Prefix for output CSV files
    """
    arrays = get_simulation_arrays(session, simulation_result_id)
    
    # Get basic info about this simulation
    info_query = text("""
        SELECT tripsum_id, section_id, status
        FROM postrun_results 
        WHERE id = :id
    """)
    result = session.execute(info_query, {"id": simulation_result_id})
    info = result.fetchone()
    
    if not info:
        print(f"No simulation result found with ID {simulation_result_id}")
        return
    
    print(f"Exporting arrays for trip {info.tripsum_id}, section {info.section_id}")
    
    # Export each array type to a separate CSV
    for array_type, array_info in arrays.items():
        filename = f"{output_prefix}_trip{info.tripsum_id}_section{info.section_id}_{array_type}.csv"
        
        df = pd.DataFrame({
            'index': range(len(array_info['values'])),
            array_type: array_info['values'],
            'units': [array_info['units']] * len(array_info['values'])
        })
        
        df.to_csv(filename, index=False)
        print(f"  Exported {array_type} ({array_info['length']} values) to {filename}")

def plot_coupler_forces(session, simulation_result_id):
    """
    Create a simple plot of coupler forces for a simulation result.
    Requires matplotlib to be installed.
    
    Args:
        session: SQLAlchemy session
        simulation_result_id: ID of the simulation result
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    # Get arrays
    arrays = get_simulation_arrays(session, simulation_result_id)
    
    # Get time and distance arrays
    time_data = arrays.get('time', {}).get('values', [])
    distance_data = arrays.get('distance', {}).get('values', [])
    
    # Get coupler force arrays
    force_arrays = {k: v for k, v in arrays.items() if k.startswith('coupler_forces_group_')}
    
    if not force_arrays:
        print("No coupler force data found")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot vs time
    if time_data:
        for array_name, array_info in force_arrays.items():
            group_num = array_name.split('_')[-1]
            ax1.plot(time_data[:len(array_info['values'])], array_info['values'], 
                    label=f'Car Group {group_num}')
        ax1.set_ylabel('Coupler Force (lbf)')
        ax1.set_title('Coupler Forces vs Time')
        ax1.legend()
        ax1.grid(True)
    
    # Plot vs distance
    if distance_data:
        for array_name, array_info in force_arrays.items():
            group_num = array_name.split('_')[-1]
            ax2.plot(distance_data[:len(array_info['values'])], array_info['values'], 
                    label=f'Car Group {group_num}')
        ax2.set_xlabel('Distance (miles)')
        ax2.set_ylabel('Coupler Force (lbf)')
        ax2.set_title('Coupler Forces vs Distance')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Example usage of the query functions.
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Query simulation results database')
    parser.add_argument('--trip', type=int, help='Trip ID to query')
    parser.add_argument('--list', action='store_true', help='List simulation results')
    parser.add_argument('--arrays', type=int, help='Get arrays for simulation result ID')
    parser.add_argument('--export', type=int, help='Export arrays to CSV for simulation result ID')
    parser.add_argument('--plot', type=int, help='Plot coupler forces for simulation result ID')
    parser.add_argument('--summary', type=int, help='Get trip summary for trip ID')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Get database connection
    engine, session = get_database_connection()
    
    try:
        if args.list:
            print("=== Simulation Results ===")
            df = list_simulation_results(session, tripsum_id=args.trip)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("No results found")
        
        if args.arrays:
            print(f"=== Arrays for Simulation Result {args.arrays} ===")
            arrays = get_simulation_arrays(session, args.arrays)
            for array_type, array_info in arrays.items():
                print(f"{array_type}: {array_info['length']} values ({array_info['units']}) - {array_info['description']}")
        
        if args.export:
            export_arrays_to_csv(session, args.export, "simulation_export")
        
        if args.plot:
            plot_coupler_forces(session, args.plot)
        
        if args.summary:
            print(f"=== Trip Summary for {args.summary} ===")
            summary = get_trip_summary(session, args.summary)
            for key, value in summary.items():
                print(f"{key}: {value}")
    
    finally:
        session.close()

if __name__ == "__main__":
    main()