#!/usr/bin/env python3
"""
Simple test script for FastSimM function using CSV data.
"""

import numpy as np
import os
import sys

# Add parent directory to path to import FastSimM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FastSimM import FastSimM

def main():
    """Test FastSimM with CSV data using specified parameters."""
    
    print("Loading test data...")
    
    # Load CSV data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    T = np.loadtxt(os.path.join(data_dir, "T.csv"), delimiter=',')
    Frope = np.loadtxt(os.path.join(data_dir, "Frope.csv"), delimiter=',')
    M = np.loadtxt(os.path.join(data_dir, "M.csv"), delimiter=',')
    b = np.loadtxt(os.path.join(data_dir, "b.csv"), delimiter=',')
    c = np.loadtxt(os.path.join(data_dir, "c.csv"), delimiter=',')
    x2all = np.loadtxt(os.path.join(data_dir, "x2all.csv"), delimiter=',')
    Fmax2all = np.loadtxt(os.path.join(data_dir, "Fmax2all.csv"), delimiter=',')
    Fmin2all = np.loadtxt(os.path.join(data_dir, "Fmin2all.csv"), delimiter=',')
    Klocked = np.loadtxt(os.path.join(data_dir, "Klocked.csv"), delimiter=',')
    Kawu = np.loadtxt(os.path.join(data_dir, "Kawu.csv"), delimiter=',')
    
    # Test parameters
    n = 28
    tstart = 0
    tend = 14224
    X0 = 'steady-state'
    
    print(f"Running FastSimM with n={n}, tstart={tstart}, tend={tend}, X0='{X0}'...")
    
    # Run FastSimM
    tode, Xode, stats, Yode = FastSimM(
        T=T, Frope=Frope, tstart=tstart, tend=tend, n=n, b=b, c=c, M=M,
        x2all=x2all, Fmax2all=Fmax2all, Fmin2all=Fmin2all, 
        Klocked=Klocked, Kawu=Kawu, X0=X0
    )
    
    print("FastSimM completed successfully!")
    print(f"Stats: {stats}")
    print(f"Output shapes: tode{tode.shape}, Xode{Xode.shape}, Yode{Yode.shape}")
    
    # Extract coupler forces (first n columns of Yode)
    forces = Yode[:, 0:n]
    print(f"Force range: [{np.min(forces):.1f}, {np.max(forces):.1f}] N")

if __name__ == "__main__":
    main()