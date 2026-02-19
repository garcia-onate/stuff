-- SQL script to create simulation results storage tables
-- Run this script to create the necessary tables in your PostgreSQL database

-- Table for simulation metadata and summary statistics
CREATE TABLE IF NOT EXISTS postrun_results (
    id SERIAL PRIMARY KEY,
    tripsum_id INTEGER NOT NULL,
    section_id INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Section metadata
    start_idx INTEGER,
    end_idx INTEGER,
    first_timestamp TIMESTAMP WITH TIME ZONE,
    sub_trip_id INTEGER,
    
    -- Section characteristics
    duration_seconds REAL,
    distance_miles REAL,
    avg_speed_mph REAL,
    max_speed_mph REAL,
    
    -- Train configuration
    train_cars INTEGER,
    train_length REAL,
    num_locomotives INTEGER,
    
    -- Simulation parameters
    dcar INTEGER,
    ts_sample REAL,
    dx_step REAL,
    x0_initial VARCHAR(50),
    
    -- Simulation results summary
    simulation_time_seconds REAL,
    total_time_steps INTEGER,
    start_distance_miles REAL,
    end_distance_miles REAL,
    num_car_groups INTEGER,
    
    -- Performance metrics (calculated from results)
    max_coupler_force_lbf REAL,
    min_coupler_force_lbf REAL,
    max_speed_achieved_mph REAL,
    total_energy_mwh REAL,
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed', -- 'completed', 'failed', 'partial'
    error_message TEXT,
    
    UNIQUE(tripsum_id, section_id)
);

-- Table for detailed time-series arrays
CREATE TABLE IF NOT EXISTS postrun_arrays (
    id SERIAL PRIMARY KEY,
    postrun_result_id INTEGER REFERENCES postrun_results(id) ON DELETE CASCADE,
    array_type VARCHAR(50) NOT NULL, -- 'time', 'distance', 'speed', 'coupler_forces', etc.
    array_data JSONB NOT NULL,
    array_length INTEGER,
    data_type VARCHAR(20), -- 'float64', 'int32', etc.
    description TEXT,
    
    UNIQUE(postrun_result_id, array_type)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_postrun_results_tripsum_id ON postrun_results(tripsum_id);
CREATE INDEX IF NOT EXISTS idx_postrun_results_created_at ON postrun_results(created_at);
CREATE INDEX IF NOT EXISTS idx_postrun_results_status ON postrun_results(status);

CREATE INDEX IF NOT EXISTS idx_postrun_arrays_result_id ON postrun_arrays(postrun_result_id);  
CREATE INDEX IF NOT EXISTS idx_postrun_arrays_type ON postrun_arrays(array_type);

-- Add comments for documentation
COMMENT ON TABLE postrun_results IS 'Stores metadata and summary statistics for train simulation results';
COMMENT ON TABLE postrun_arrays IS 'Stores detailed time-series arrays from train simulations as JSONB';

COMMENT ON COLUMN postrun_results.tripsum_id IS 'Foreign key to trip_summary table';
COMMENT ON COLUMN postrun_results.section_id IS 'Sequential ID for sections within a trip';
COMMENT ON COLUMN postrun_results.status IS 'Simulation status: completed, failed, or partial';

COMMENT ON COLUMN postrun_arrays.array_type IS 'Type of array data: time, distance, speed, coupler_forces, etc.';
COMMENT ON COLUMN postrun_arrays.array_data IS 'JSONB containing array values and metadata';