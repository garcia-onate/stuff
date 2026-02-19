"""
Command-line interface for PySim package
"""

import argparse
import sys
import os
import json

def main_run():
    """Main CLI entry point for running simulations"""
    parser = argparse.ArgumentParser(description='Run PySim train simulation')
    parser.add_argument('--profile', required=True, help='Profile JSON file')
    parser.add_argument('--spec', required=True, help='Spec JSON file')
    parser.add_argument('--track', required=True, help='Track data JSON file')
    parser.add_argument('--train', required=True, help='Train data JSON file')
    parser.add_argument('--locos', required=True, help='Locomotives JSON file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--dcar', type=int, help='Number of cars per group')
    parser.add_argument('--ts', type=float, help='Time step')
    parser.add_argument('--dx', type=float, help='Distance step')
    
    args = parser.parse_args()
    
    try:
        from pysim import runSim
        from pysim.runPostRun import load_json_data
        
        # Load input data
        profile = load_json_data(args.profile)
        spec = load_json_data(args.spec)
        track_data = load_json_data(args.track)
        train = load_json_data(args.train)
        locos = load_json_data(args.locos)
        
        # Run simulation
        results = runSim(profile, spec, track_data, train, locos, 
                        dcar=args.dcar, Ts=args.ts, dx=args.dx)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print("Simulation completed successfully")
            
    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)

def main_postrun():
    """Main CLI entry point for post-run analysis"""
    parser = argparse.ArgumentParser(description='Run PySim post-run analysis')
    parser.add_argument('--tripsum-id', type=int, required=True, help='Trip summary ID')
    parser.add_argument('--dcar', type=int, help='Number of cars per group')
    parser.add_argument('--ts', type=float, help='Time step')
    parser.add_argument('--dx', type=float, help='Distance step')
    parser.add_argument('--csv', help='CSV file for trip data')
    parser.add_argument('--no-db', action='store_true', help='Do not store results in database')
    parser.add_argument('--no-csv', action='store_true', help='Do not save CSV files')
    
    args = parser.parse_args()
    
    try:
        from pysim.runPostRun import run_post_run_analysis
        
        run_post_run_analysis(
            tripsum_id=args.tripsum_id,
            dcar=args.dcar,
            Ts=args.ts,
            dx=args.dx,
            csv_file=args.csv,
            store_in_db=not args.no_db,
            save_csv=not args.no_csv
        )
        
    except Exception as e:
        print(f"Error running post-run analysis: {e}")
        sys.exit(1)

def main_analysis():
    """Main CLI entry point for analysis tools"""
    parser = argparse.ArgumentParser(description='PySim analysis tools')
    parser.add_argument('--list-trips', action='store_true', help='List available trips')
    parser.add_argument('--validate', help='Validate input data files')
    
    args = parser.parse_args()
    
    try:
        if args.list_trips:
            from pysim.runPostRun import get_database_connection, list_available_trips
            engine, session = get_database_connection()
            list_available_trips(session)
            session.close()
            
        elif args.validate:
            from pysim.runPostRun import load_json_data
            data = load_json_data(args.validate)
            if data:
                print(f"File {args.validate} is valid JSON")
            else:
                print(f"File {args.validate} is not valid JSON")
                sys.exit(1)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.argv.pop(1)
        main_run()
    elif len(sys.argv) > 1 and sys.argv[1] == "postrun":
        sys.argv.pop(1)
        main_postrun()
    elif len(sys.argv) > 1 and sys.argv[1] == "analysis":
        sys.argv.pop(1)
        main_analysis()
    else:
        print("Usage: python -m pysim.cli {run|postrun|analysis} [options]")
        sys.exit(1)