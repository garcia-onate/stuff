# FastSim Data Structure Documentation

This document describes the key data structures used in the FastSim train dynamics simulation system, based on analysis of the MATLAB codebase.

## 1. Track Dataset Structure (`dataset`)

Returned by `getfulltrackstruc(name)` - Contains all track-related information.

### Main Fields:

#### `dataset.Grade`
Track elevation and geometry data:
- **`Dist`** *(array)*: Distance points along track (miles)
- **`Grade`** *(array)*: Track grade at each distance point (percent)
- **`Curvature`** *(array)*: Track curvature at each point (degrees/mile) - *auto-filled with zeros if missing*
- **`SuperElev`** *(array)*: Super-elevation at each point (inches) - *auto-filled with zeros if missing*

#### `dataset.SpdLim`
Speed limit information:
- **`Dist`** *(array)*: Distance points for speed limits (miles)
- **`Limit`** *(array)*: Speed limit at each point (mph)
- **`Type`** *(array)*: Speed limit type code - *auto-filled with zeros if missing*

#### `dataset.GPS`
Geographic positioning data:
- **`Dist`** *(array)*: Distance points (miles)
- **`Nlat`** *(array)*: North latitude coordinates
- **`Elon`** *(array)*: East longitude coordinates
- *Note: Auto-generated as `[0, track_end]` if missing*

#### `dataset.Milepost`
Milepost mapping information:
- **`Dist`** *(array)*: Distance points (miles)
- **`MP`** *(array)*: Corresponding milepost values
- *Note: Auto-generated as linear mapping if missing*

#### `dataset.PathDist`
- **`PathDist`** *(scalar)*: Total path distance of track segment (miles)

#### `dataset.PLM` *(optional)*
Power Limit Management data (if present in track file):
- Structure containing power limit information along the track

---

## 2. Train Parameter Structure (`trainParam` / `spec.Train`)

Contains all train composition and characteristics data.

### Physical Properties:

#### **`NumCars`** *(scalar)*
- Number of freight cars in the train (excludes locomotives)
- Example: `132`

#### **`Length`** *(scalar)*
- Total train length in feet
- Example: `7484.0`

#### **`LocoPosition`** *(array)*
- Positions of locomotives in the train consist (1-indexed)
- Example: `[1, 2, 135, 136]` (head-end and mid-train helpers)

### Car-Level Data Arrays:
*All arrays have length = `NumCars`*

#### **`LoadWeight`** *(array)*
- Weight of each car in tons
- Can be scalar (distributed equally) or per-car array
- Example: `[140, 141, 51, 51, 141, ...]` (132 values)

#### **`LoadLength`** *(array)*
- Length of each car in feet
- Auto-calculated if missing: `(Length - loco_length) / NumCars`
- Example: `[46, 42, 69, 69, 42, ...]` (132 values)

#### **`CouplerType`** *(array)*
- Coupler type for each car position
- `0` = Standard draft gear
- `3` = EOCC (End-of-Car Cushioning)
- Processed as: `(CouplerType == 0) + (CouplerType == 3)*2`

#### **`PreLoad`** *(array)*
- Coupler preload force in pounds (for EOCC couplers)
- Set to `0` for draft gears automatically
- Typical values: `50000`, `100000` pounds
- Example: `[0, 0, 100000, 50000, ...]` (132 values)

#### **`Stroke`** *(array)* *(optional)*
- EOCC stroke length in inches
- Default: `28` inches for EOCC, `0` for draft gears
- Auto-generated if missing

### Train-Level Drag Coefficients:

#### **`Davis_a`** *(scalar)*
- Davis equation constant term (lbf)
- Example: `1.32032134`

#### **`Davis_b`** *(scalar)*
- Davis equation linear velocity coefficient (lbf per mph)  
- Example: `0.01100785`

#### **`Davis_c`** *(scalar)*
- Davis equation quadratic velocity coefficient (lbf per mph²)
- Example: `0.000591`

### Computed Fields:
*Added by `optspec_dtm()` processing*

#### **`Weight`** *(scalar)*
- Total train weight: `sum(loco_weights) + sum(LoadWeight)`

#### **`GrossWeight`** *(array)*
- Gross weight per car (for DTM calculations)
- Computed based on car length and loading rules

#### **`TareWeight`** *(array)*
- Tare (empty) weight per car
- Computed as `GrossWeight / 4.2`

---

## 3. Locomotive Parameter Structure (`locoParam`)

Array of locomotive structures, one per locomotive.

### Fields per Locomotive:

#### **`Weight`** *(scalar)*
- Locomotive weight in tons
- Example: `207` (for typical road locomotive)

#### **`Length`** *(scalar)*
- Locomotive length in feet
- Default: `76` feet if missing
- Example: `73.2`

#### **`Power`** *(scalar)*
- Rated power in horsepower
- Example: `4400`

#### **`MaxTE`** *(array)*
- Maximum tractive effort by notch (lbf)
- Length matches `Notch` array
- Sign automatically matched to `Notch` sign

#### **`Notch`** *(array)*
- Throttle notch positions
- Positive for traction, negative for braking
- Example: `[-8, -7, -6, ..., 0, 1, 2, ..., 8]`

#### **`Speed`** *(array)*
- Speed points for tractive effort curves (mph)
- Used with `MaxTE` for effort vs. speed lookup

---

## 4. Trip Parameter Structure (`spec.Trip`)

Defines the operating scenario for the simulation.

### Route Definition:

#### **`StartDist`** *(scalar)*
- Starting distance on track (miles)
- Example: `98.0`

#### **`EndDist`** *(scalar)*
- Ending distance on track (miles)  
- Example: `259.0`

#### **`StartSpeed`** *(scalar)*
- Initial train speed (mph)
- Example: `30.0`

#### **`EndSpeed`** *(scalar)*
- Target final speed (mph)
- Example: `30.0`

### Power Limit Management:

#### **`PLM`** *(structure)*
- **`Dist`** *(array)*: Distance points for power limit changes
- **`Notch`** *(array)*: Required notch settings at each point
- Example: `Dist: [0, 0, 170, 175]`, `Notch: [-8, 8, 4, 8]`

### Slow Orders:

#### **`SlowOrders`** *(structure)*
- **`StartDist`** *(array)*: Start distances for temporary speed restrictions
- **`EndDist`** *(array)*: End distances for speed restrictions  
- **`Mph`** *(array)*: Speed limit during restriction (mph)
- **`Type`** *(array)*: Restriction type code
- **`OverrideLimit`** *(array)*: Whether to override track speed limit

#### **`additionalMinutes`** *(scalar)*
- Additional time allowance for schedule (minutes)
- Can be `NaN` if not specified

---

## 5. Simulation Configuration

### **`spec.Route`** *(scalar or string)*
- Reference to track database entry
- Example: `"UP_PineBluff_HB"`

### **`spec.Consist`** *(array)*
- Array of locomotive type references
- Used by `getlocostruc()` to load locomotive parameters
- Example: `[reference1, reference2, reference3, reference4]`

### **`spec.TFSsave`** *(scalar)*
- Time step for saving results (seconds)
- Example: `5.0`

### Additional Control Parameters:

#### **`spec.MaxPositionError`** *(scalar)*
- Maximum allowable position error (feet)
- Example: `500.0`

#### **`spec.MaxRandomMassError`** *(scalar)*
- Maximum random mass error percentage
- Example: `10.0`

#### **`spec.MaxRandomMassDistributionError`** *(scalar)*
- Maximum random mass distribution error percentage  
- Example: `10.0`

#### **`spec.MeanPositionError`** *(scalar)*
- Mean position error (feet)
- Example: `0.0`

---

## Usage Patterns

### Track Data Loading:
```matlab
dataset = getfulltrackstruc(spec.Route);
% Access: dataset.Grade.Dist, dataset.SpdLim.Limit, etc.
```

### Train Data Processing:
```matlab
trainParam = spec.Train;
locoParam = getlocostruc(spec.Consist);
spec = optspec_dtm(spec);  % Adds GrossWeight, TareWeight fields
```

### Force Calculations:
```matlab
F = gravityforces(dataset, trainParam, locoParam, distout);
% Uses: trainParam.LoadWeight, trainParam.LoadLength, trainParam.LocoPosition
```

### Physics Simulation:
```matlab
[Klocked, Kawu, groupMassVector, ...] = initFastSim(spec, dcar, Ts);
% Uses: trainParam.CouplerType, trainParam.PreLoad, trainParam.Stroke
```

This structure provides a complete specification of train consists, track geometry, operating constraints, and simulation parameters needed for the FastSim physics engine.