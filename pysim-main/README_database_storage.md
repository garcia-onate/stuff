# Simulation Results Database Storage

This document explains the new database storage functionality for train simulation results.

## Overview

The enhanced `runPostRun.py` script now stores simulation results in PostgreSQL tables, enabling efficient querying and visualization through web interfaces. The system uses a two-table design:

1. **`postrun_results`** - Stores metadata and summary statistics
2. **`postrun_arrays`** - Stores detailed time-series data as JSONB

## Database Setup

### 1. Create Tables

Run the SQL script to create the necessary tables:

```bash
psql -h your-host -U your-user -d your-database -f create_simulation_tables.sql
```

### 2. Verify Tables

Check that the tables were created successfully:

```sql
\dt postrun_*
SELECT COUNT(*) FROM postrun_results;
SELECT COUNT(*) FROM postrun_arrays;
```

## Usage

### Running Simulations with Database Storage

The `runPostRun.py` script now includes database storage by default:

```bash
# Basic usage (stores results in database)
python runPostRun.py 535

# Load from CSV and store in database
python runPostRun.py 535 --csv trip_data_535.csv

# Clear existing results before storing new ones
python runPostRun.py 535 --clear-existing

# Disable database storage (original behavior)
python runPostRun.py 535 --no-db-storage

# Advanced options
python runPostRun.py 535 --dcar 10 --ts 1.0 --speed-threshold 20 --clear-existing
```

### Command Line Options

New database-related options:

- `--no-db-storage`: Disable database storage of results
- `--clear-existing`: Clear existing results for this trip before storing new ones

### Querying Results

Use the `query_simulation_results.py` script to query stored results:

```bash
# List all simulation results
python query_simulation_results.py --list

# List results for a specific trip
python query_simulation_results.py --list --trip 535

# Get trip summary
python query_simulation_results.py --summary 535

# Get arrays for a specific simulation result
python query_simulation_results.py --arrays 123

# Export arrays to CSV files
python query_simulation_results.py --export 123

# Plot coupler forces (requires matplotlib)
python query_simulation_results.py --plot 123
```

## Database Schema

### postrun_results Table

Stores simulation metadata and summary statistics:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| tripsum_id | INTEGER | Trip summary ID |
| section_id | INTEGER | Section number within trip |
| created_at | TIMESTAMP | When simulation was run |
| status | VARCHAR(20) | 'completed', 'failed', or 'partial' |
| duration_seconds | REAL | Section duration |
| distance_miles | REAL | Section distance |
| train_cars | INTEGER | Number of cars |
| simulation_time_seconds | REAL | Time to run simulation |
| max_coupler_force_lbf | REAL | Maximum coupler force |
| error_message | TEXT | Error message for failed simulations |

### postrun_arrays Table

Stores detailed time-series arrays as JSONB:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| postrun_result_id | INTEGER | Foreign key to postrun_results |
| array_type | VARCHAR(50) | Type: 'time', 'distance', 'speed', etc. |
| array_data | JSONB | Array values and metadata |
| array_length | INTEGER | Number of elements |
| description | TEXT | Human-readable description |

### Array Data Format

Arrays are stored as JSONB with this structure:

```json
{
  "values": [0.0, 0.5, 1.0, 1.5, ...],
  "units": "mph",
  "length": 1000,
  "data_type": "float64",
  "description": "Speed data for car group 1"
}
```

## Array Types Stored

For each simulation section, the following arrays are stored:

- `time` - Time vector in hours
- `distance` - Distance vector in miles
- `speed_group_N` - Speed for each car group (mph)
- `coupler_forces_group_N` - Coupler forces for each car group (lbf)
- `acceleration_group_N` - Acceleration for each car group (mph/s)

## Flask API Integration

The stored data is designed for easy integration with Flask APIs:

### Suggested API Endpoints

```python
# Get trip overview
GET /api/trips/{tripsum_id}/simulations

# Get specific section results  
GET /api/trips/{tripsum_id}/sections/{section_id}/results

# Get array data for plotting
GET /api/trips/{tripsum_id}/sections/{section_id}/arrays/{array_type}

# Get force analysis summary
GET /api/trips/{tripsum_id}/forces/summary
```

### Example Flask Route

```python
@app.route('/api/trips/<int:tripsum_id>/simulations')
def get_trip_simulations(tripsum_id):
    query = text("""
        SELECT id, section_id, status, distance_miles, 
               max_coupler_force_lbf, created_at
        FROM postrun_results 
        WHERE tripsum_id = :tripsum_id
        ORDER BY section_id
    """)
    
    results = session.execute(query, {"tripsum_id": tripsum_id})
    return jsonify([dict(row) for row in results])
```

## Testing

Test the database storage functionality:

```bash
# Run storage tests (creates and cleans up test data)
python test_db_storage.py

# Test with actual trip data
python runPostRun.py 535 --csv trip_data_535.csv --clear-existing

# Query the results
python query_simulation_results.py --list --trip 535
```

## Performance Considerations

1. **Indexing**: The schema includes indexes on frequently queried columns
2. **JSONB**: PostgreSQL's JSONB format provides efficient storage and querying of array data
3. **Chunking**: For very large simulations, consider splitting long arrays into chunks
4. **Cleanup**: Use `--clear-existing` to avoid duplicate data when re-running simulations

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify database credentials and network connectivity
2. **Table Missing**: Run `create_simulation_tables.sql` to create required tables
3. **Duplicate Data**: Use `--clear-existing` when re-running simulations
4. **Large Arrays**: Monitor JSONB storage size for very long simulations

### Debugging

Enable detailed logging by modifying the scripts to print SQL queries and array sizes.

## Future Enhancements

Potential improvements:

1. **Compression**: Compress large arrays before storage
2. **Partitioning**: Partition tables by date or trip ID for better performance  
3. **Caching**: Add Redis caching for frequently accessed arrays
4. **Streaming**: Implement streaming for very large datasets
5. **Validation**: Add data validation and checksums