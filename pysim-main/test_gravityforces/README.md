# Gravity Forces Test Documentation

## Overview

This directory contains a comprehensive test for the `gravityforces.py` function, which computes gravity forces on a train at the end of each car based on track grade and train characteristics.

## Files

- `test_gravityforces.py` - Main test script
- `dataset.txt` - Track dataset (JSON format) containing grade, speed limits, GPS coordinates, etc.
- `trainParam.txt` - Train parameters (JSON format) including car weights, lengths, locomotive positions
- `locoParam.txt` - Locomotive parameters (JSON format) including locomotive specifications
- `distout.csv` - Distance points along the track (CSV format) where forces are calculated
- `gravityforces_output.csv` - Generated output file with calculated forces

## Running the Test

```bash
cd test_gravityforces
python test_gravityforces.py
```

Or make it executable and run directly:
```bash
chmod +x test_gravityforces.py
./test_gravityforces.py
```

## Data Structures

### Dataset Structure
Based on `fastsim_data_structures.md`, the dataset contains:
- **Grade**: Track elevation and geometry data
  - `Dist`: Distance points along track (miles)
  - `Percent`: Track grade at each point (percent)
  - `Curvature`: Track curvature (degrees/mile)
- **PathDist**: Total path distance (scalar, miles)
- **SpdLim**, **GPS**, **Milepost**: Additional track information

### TrainParam Structure
Train composition and characteristics:
- **LoadWeight**: Weight of each car (tons)
- **LoadLength**: Length of each car (feet)
- **LocoPosition**: Positions of locomotives in consist (1-indexed)
- **NumCars**: Number of freight cars (excludes locomotives)

### LocoParam Structure
Array of locomotive specifications:
- **Weight**: Locomotive weight (tons)
- **Length**: Locomotive length (feet)
- Additional locomotive parameters (power, tractive effort, etc.)

## Expected Output

The `gravityforces` function returns:
- **F**: Matrix of gravity forces (lbf)
  - Shape: `(len(distout), NumCars + len(locoParam))`
  - Each row corresponds to a distance point
  - Each column corresponds to a car/locomotive unit
  - Forces are in pounds (lbf), negative values indicate forces opposing motion

## Test Results

The test script validates:
1. **Data Loading**: All input files load correctly
2. **Data Structure**: Input data matches expected FastSim format
3. **Function Execution**: `gravityforces` runs without errors
4. **Output Validation**: Results have expected shape and reasonable values
5. **Force Analysis**: Statistical summary of calculated forces

## Sample Output

```
=== Testing gravityforces function ===
Calling gravityforces with:
  - Train cars: 132
  - Locomotives: 4
  - Distance points: 1611
  - Track distance range: [98.0, 259.0] miles

✓ gravityforces completed successfully!
Output shape: (1611, 136)
Expected shape: (1611, 136)

Force analysis:
  Force range: [-7837.0, 5775.3] lbf
  Mean absolute force: 1086.2 lbf
  Standard deviation: 1450.4 lbf
✓ Force magnitudes appear reasonable
```

## Notes

- Forces are calculated based on track grade and train mass distribution
- Positive forces assist motion (downhill), negative forces oppose motion (uphill)
- The function uses `makeeffelev` to compute effective elevation from grade data
- Results are saved to `gravityforces_output.csv` for further analysis