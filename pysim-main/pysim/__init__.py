"""
PySim - Python Train Simulation Package

A comprehensive Python package for train dynamics simulation using the FastSim methodology.
Originally developed from MATLAB implementations by H. Kirk Mathews (GE Research).

This package provides:
- Complete train dynamics simulation capabilities
- In-train force modeling and analysis
- Database integration for operational data
- Performance analysis and visualization tools
- Comprehensive testing and validation suites

Usage:
    from pysim import runSim
    
    # Run complete simulation
    results = runSim(profile, spec, track_data, train, locos)
    
    # Or use individual components
    from pysim.runFastSimM import runFastSimM
    from pysim.calcLVratio import calcLVratio
    from pysim.runPostRun import run_post_run_analysis

Version: 1.0.0
Author: Joseph Wakeman (Wabtec Corporation)
Original MATLAB: H. Kirk Mathews (GE Research)
"""

__version__ = "1.0.0"
__author__ = "Joseph Wakeman"
__email__ = "joseph.wakeman@wabtec.com"

# Import all the main modules with their original names
try:
    from .runSim import runSim
    from .runFastSimM import runFastSimM
    from .FastSimM import FastSimM
    from .FastSimODE import FastSimODE
    from .initFastSim import initFastSim
    from .calcRopeForces import calcRopeForces
    from .calcLVratio import calcLVratio
    from .interpFastSim import interpFastSim
    from .downSampleFastSim import downSampleFastSim
    from .normalizeProfile import normalizeProfile
    from .gravityforces import gravityforces
    from .locoeffort_of_notch_speed import locoeffort_of_notch_speed
    from .defineCarGroups import defineCarGroups
    from .defineCouplers import defineCouplers
    from .expandSpec import expandSpec
    from .makeeffelev import makeeffelev
    from .optspec_dtm import optspec_dtm
    from .slewRateFilter import slewRateFilter
    from .ssCouplerDisplacement import ssCouplerDisplacement
    from .ode23simple import ode23simple
    from .unit_conversions import *
    from .models import Base, TripSummary, DataRecorder, ComplianceTrip
    from .runPostRun import run_post_run_analysis
except ImportError as e:
    # Some modules might not be available or have import issues
    # This allows the package to still be importable for basic functionality
    pass

# Try to import upsampleProfile if it exists
try:
    from .upsampleProfile import upsampleProfile
except ImportError:
    upsampleProfile = None

__all__ = [
    # Main simulation functions - core entry points
    "runSim",              # Complete simulation pipeline
    "runFastSimM",         # Core FastSim simulation  
    "FastSimM",            # FastSim dynamics engine
    "FastSimODE",          # ODE function for FastSim
    "initFastSim",         # Simulation initialization
    
    # Analysis and calculation functions
    "calcRopeForces",      # Locomotive force calculations
    "calcLVratio",         # L/V ratio analysis
    "interpFastSim",       # Force interpolation to individual cars
    "downSampleFastSim",   # Intelligent data downsampling
    "gravityforces",       # Gravity force calculations
    "locoeffort_of_notch_speed",  # Locomotive effort curves
    
    # Profile and data processing
    "normalizeProfile",    # Profile normalization
    "expandSpec",          # Spec expansion utilities
    "makeeffelev",         # Effective elevation calculation
    "optspec_dtm",         # DTM optimization
    "slewRateFilter",      # Slew rate filtering
    
    # Configuration and setup
    "defineCarGroups",     # Car grouping for efficiency
    "defineCouplers",      # Coupler force tables
    "ssCouplerDisplacement", # Steady-state displacement
    
    # Utilities
    "ode23simple",         # ODE solver
    
    # Database and post-processing  
    "run_post_run_analysis", # Database analysis workflow
    "Base", "TripSummary", "DataRecorder", "ComplianceTrip", # Database models
]