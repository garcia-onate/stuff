"""Device management utilities for PyTorch.

Centralized device selection logic for GPU/CPU operations.
"""

import torch


def get_device(preference=None):
    """Get PyTorch device for computation.
    
    Parameters
    ----------
    preference : str, optional
        Device preference ('cuda', 'cpu', or None for auto-detect)
        
    Returns
    -------
    torch.device
        PyTorch device object
        
    Examples
    --------
    >>> device = get_device()  # Auto-detect
    >>> device = get_device('cuda')  # Force CUDA
    >>> device = get_device('cpu')  # Force CPU
    """
    if preference == 'cpu':
        return torch.device('cpu')
    elif preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            print("Warning: CUDA requested but not available, using CPU")
            return torch.device('cpu')
    else:  # Auto-detect
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_device_name(device):
    """Get human-readable device name.
    
    Parameters
    ----------
    device : torch.device
        PyTorch device
        
    Returns
    -------
    str
        Device name
    """
    if device.type == 'cuda':
        return f"CUDA: {torch.cuda.get_device_name(device.index)}"
    else:
        return "CPU"
