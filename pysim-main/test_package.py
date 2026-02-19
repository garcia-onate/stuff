#!/usr/bin/env python3
"""
Test script to verify the pysim package is properly installed and importable
"""

print("Testing pysim package import...")

try:
    import pysim
    print("✓ Package imported successfully!")
    
    print(f"✓ Package version: {pysim.__version__}")
    print(f"✓ Package author: {pysim.__author__}")
    
    # Test importing key functions
    from pysim import runSim, runFastSimM, calcLVratio
    print("✓ Key functions imported successfully!")
    
    # Show available functions
    available_funcs = [f for f in dir(pysim) if not f.startswith('_')]
    print(f"✓ Available functions ({len(available_funcs)} total):")
    for i, func in enumerate(available_funcs[:15]):  # Show first 15
        print(f"  - {func}")
    if len(available_funcs) > 15:
        print(f"  ... and {len(available_funcs) - 15} more")
    
    print("\n✓ PySim package is ready to use!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")