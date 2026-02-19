# PySim - Python Train Simulation Package

A comprehensive Python package for train dynamics simulation using the FastSim methodology. Originally developed from MATLAB implementations by H. Kirk Mathews (GE Research).

## Overview

PySim provides a complete simulation environment for modeling train dynamics, in-train forces, and operational analysis. The package is designed for railway engineering applications, offering:

- **Complete train dynamics simulation** using the FastSim methodology
- **In-train force modeling** with detailed coupler and car interactions  
- **Database integration** for operational data analysis
- **Performance analysis tools** for derailment assessment and optimization
- **Comprehensive testing framework** with validation suites

## Key Features

### Core Simulation Engine
- **FastSim dynamics**: Advanced train simulation with grouped car modeling
- **Physics-based calculations**: Rope forces, gravity effects, locomotive effort
- **Flexible initialization**: Support for steady-state and custom initial conditions
- **Adaptive time stepping**: Efficient ODE integration with error control

### Analysis Capabilities
- **L/V ratio calculation**: Lateral-to-vertical force analysis for derailment assessment
- **Force interpolation**: Individual car force calculation from grouped results
- **Data downsampling**: Intelligent sampling for efficient storage and analysis
- **Performance metrics**: Comprehensive simulation result analysis

### Database Integration
- **PostgreSQL connectivity**: Direct integration with operational databases
- **Trip data processing**: Automated analysis of recorded train operations
- **Result storage**: Structured storage of simulation results and metadata
- **CSV export**: Flexible data export capabilities

## Installation

### From Source
```bash
git clone https://gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git
cd pysim
pip install -e .
```

### Dependencies
The package requires Python 3.8+ and the following key dependencies:
- numpy, scipy - Numerical computing
- pandas - Data manipulation
- sqlalchemy - Database integration
- matplotlib - Visualization
- jupyter - Interactive analysis

See `requirements.txt` for complete dependency list.

## Quick Start

### Basic Simulation
```python
from pysim import runSim

# Load your data (profile, spec, track data, train config, locomotives)
results = runSim(profile, spec, track_data, train, locos)

# Access results
forces = results['Fsi_allcars']  # Individual car forces
lv_ratios = results['LVratio']   # L/V ratios
distances = results['dist']      # Distance vector
```

### Individual Components
```python
from pysim import runFastSimM, calcLVratio, interpFastSim

# Run core simulation
fs = runFastSimM(profile, spec, track_data, train, locos, dcar=5)

# Interpolate to individual cars
fs = interpFastSim(fs, spec, track_data, locos)

# Calculate L/V ratios
lv_ratios = calcLVratio(spec, fs['Fsi_allcars'], fs['dist'])
```

### Database Analysis
```python
from pysim.runPostRun import run_post_run_analysis

# Analyze recorded trip data
run_post_run_analysis(
    tripsum_id=12345,
    dcar=5,
    store_in_db=True,
    save_csv=True
)
```

### Command Line Interface
```bash
# Run simulation from command line
pysim-run --profile profile.json --spec spec.json --track track.json --train train.json --locos locos.json

# Post-run analysis
pysim-postrun --tripsum-id 12345 --dcar 5

# List available trips
pysim-analysis --list-trips
```

## Project Structure

```
pysim/
├── pysim/                 # Main package
│   ├── runSim.py         # Main simulation pipeline
│   ├── runFastSimM.py    # Core FastSim simulation
│   ├── FastSimM.py       # FastSim dynamics engine
│   ├── calcRopeForces.py # Force calculations
│   ├── calcLVratio.py    # L/V ratio analysis
│   ├── models.py         # Database models
│   ├── runPostRun.py     # Post-run database analysis
│   └── ...               # Additional modules
├── test_*/               # Test suites
├── docs/                 # Documentation
├── setup.py             # Package setup
└── requirements.txt     # Dependencies
```

## Testing

The package includes comprehensive test suites:

```bash
# Run specific component tests
cd test_runFastSimM && python test_runFastSimM.py
cd test_initFastSim && python test_initFastSim.py

# Run complete pipeline test
cd test_runFastSimM && python test_runSim.py
```

## Documentation

- **Code flow diagrams**: See `docs/code_flow_diagrams.md`
- **Data structures**: See `docs/fastsim_data_structures.md`
- **Database storage**: See `README_database_storage.md`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a merge request

## License

Copyright (c) 2025 Wabtec Corporation. All rights reserved.

## Authors

- **Joseph Wakeman** - Python implementation (Wabtec Corporation)
- **H. Kirk Mathews** - Original MATLAB implementation (GE Research)

## Acknowledgments

This package is based on the FastSim methodology developed at GE Research for high-fidelity train dynamics simulation. The Python implementation maintains compatibility with the original MATLAB algorithms while providing enhanced database integration and analysis capabilities.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
