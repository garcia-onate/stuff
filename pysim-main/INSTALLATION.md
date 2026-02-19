# PySim Installation Guide

## Core Installation (Minimal Dependencies)

For core simulation functionality only:

```bash
# From Git (minimal dependencies)
pip install git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git

# Only installs: numpy, scipy, pandas, sqlalchemy, python-dateutil
```

## Optional Dependencies

### Visualization Support
If you need plotting capabilities:
```bash
pip install git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git[viz]
# Adds: matplotlib, plotly, seaborn
```

### Jupyter Notebook Support  
If you want to use pysim in Jupyter notebooks:
```bash
pip install git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git[jupyter]
# Adds: jupyter, ipywidgets, matplotlib
```

### Everything
If you want all optional features:
```bash
pip install git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git[all]
# Adds: matplotlib, plotly, seaborn, jupyter, ipywidgets
```

### Development
If you're developing the package:
```bash
pip install git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git[dev]
# Adds: pytest, black, flake8, mypy for development
```

## Core Dependencies (Always Installed)

- **numpy** - Numerical computing
- **scipy** - Scientific computing and ODE solving
- **pandas** - Data manipulation for database integration
- **sqlalchemy** - Database ORM for runPostRun functionality
- **python-dateutil** - Date/time handling

## Usage Examples

```python
# Core simulation (no extra dependencies needed)
from pysim import runSim, runFastSimM
from pysim.runPostRun import run_post_run_analysis

# Basic simulation
results = runSim(profile, spec, track_data, train, locos)

# Database analysis (requires sqlalchemy - included in core)
run_post_run_analysis(tripsum_id=12345)

# Plotting (requires [viz] extra)
import matplotlib.pyplot as plt
plt.plot(results['dist'], results['Fsi_allcars'])
```

This approach ensures that projects using pysim only install what they actually need!

## Updating from Previous Version

If you have a previous version of pysim installed in another project, update it with:

```bash
pip install --force-reinstall git+ssh://git@gitlab.corp.wabtec.com/Joseph.Wakeman/pysim.git
```

**Important**: Make sure to commit and push your changes to GitLab first:
```bash
cd /home/joseph.wakeman/pysim
git add .
git commit -m "Add data files and fix runPostRun path issues"
git push origin main
```

## Fixed Issues

- ✅ **runPostRun data files**: Locomotive and spec data files are now included in the package
- ✅ **Minimal dependencies**: Reduced from 115+ to 5 core dependencies  
- ✅ **Optional extras**: Visualization and Jupyter support available as extras