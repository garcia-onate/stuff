# initFastSim Test

This directory contains a test script for the `initFastSim.py` function.

## Files

- `test_initFastSim.py` - Main test script
- `spec.txt` - JSON file containing the train specification structure  
- `locoParam.txt` - JSON file containing the locomotive parameters structure

## Running the Test

Run the test from this directory:

```bash
python test_initFastSim.py
```

Or with the configured Python environment:

```bash
/home/joseph.wakeman/pysim/venv/bin/python test_initFastSim.py
```

## Test Parameters

The test uses the following parameters:
- `dcar = 5` - Number of cars grouped together for computational efficiency
- `Ts = 0.5` - Desired time step of the output

## Expected Output

The test will:
1. Load and validate input data structures
2. Call the `initFastSim` function with test parameters
3. Analyze and validate the results
4. Display comprehensive statistics about the outputs

The test validates:
- Correct data structure formats
- Reasonable physical values (positive masses, spring constants, etc.)
- No NaN or infinite values in results
- Consistent car grouping
- Force envelope construction

## Results

The function returns:
- `Klocked` - Locked coupler spring constants
- `Kawu` - Anti-windup gains  
- `groupMassVector` - Mass of each car group
- `groupDampingB` - Linear damping coefficients
- `groupCushioningDampingB` - Non-linear damping coefficients
- `cargroup` - Car group boundaries
- `x2all` - Coupler displacement vector
- `Fmin2all` - Minimum force envelopes for each coupler
- `Fmax2all` - Maximum force envelopes for each coupler