# Unit conversion functions for FastSim
# These functions provide unit conversion constants used throughout the simulation
# multiply by function_() to convert to SI units or
# use function_(x) to convert x to SI units

def lbf_(x=None):
    """Convert lbf to N (Newtons) or return conversion factor"""
    scale_lbf = 4.44822  # lbf to N
    if x is None:
        return scale_lbf
    else:
        return scale_lbf * x

def lb_(x=None):
    """Convert lb to kg or return conversion factor"""
    scale_lb = 0.453592  # lb to kg
    if x is None:
        return scale_lb
    else:
        return scale_lb * x

def kips_(x=None):
    """Convert kips to N (Newtons) or return conversion factor"""
    scale_kips = 1000 * lbf_()  # kips (i.e. kilopounds) to N
    if x is None:
        return scale_kips
    else:
        return scale_kips * x

def ton_(x=None):
    """Convert ton to kg or return conversion factor"""
    scale_ton = 2000 * lb_()  # ton (US or short ton) to kg
    if x is None:
        return scale_ton
    else:
        return scale_ton * x

def in_(x=None):
    """Convert inches to m or return conversion factor"""
    scale_in = 0.0254  # inches to m
    if x is None:
        return scale_in
    else:
        return scale_in * x

def mm_(x=None):
    """Convert mm to m or return conversion factor"""
    scale_mm = 0.001  # mm to m
    if x is None:
        return scale_mm
    else:
        return scale_mm * x

def sec_(x=None):
    """Convert sec to sec (identity) or return conversion factor"""
    scale_sec = 1.0  # sec to sec
    if x is None:
        return scale_sec
    else:
        return scale_sec * x

def minute_(x=None):
    """Convert minutes to sec or return conversion factor"""
    scale_minute = 60.0  # minute to sec
    if x is None:
        return scale_minute
    else:
        return scale_minute * x

def hour_(x=None):
    """Convert hours to sec or return conversion factor"""
    scale_hour = 3600.0  # hour to sec
    if x is None:
        return scale_hour
    else:
        return scale_hour * x