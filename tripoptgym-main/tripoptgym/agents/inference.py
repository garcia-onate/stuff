"""Model inference utilities for loading and running trained agents."""

import torch
import numpy as np

from tripoptgym.agents.network import Network
from tripoptgym.utils.device import get_device


def load_trained_agent(checkpoint_path, state_size=6, action_size=3, device=None):
    """Load a trained DQN agent from checkpoint file.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint .pth file
    state_size : int, optional
        State space dimension (default: 6)
    action_size : int, optional
        Number of actions (default: 3)
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    tuple
        (network, device) - Loaded network and device
        
    Examples
    --------
    >>> network, device = load_trained_agent('checkpoint.pth')
    >>> action = trained_agent_policy(network, device, state)
    """
    if device is None:
        device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict (handle both formats)
    if 'local_qnetwork_state_dict' in checkpoint:
        # Full checkpoint with training state
        state_dict = checkpoint['local_qnetwork_state_dict']
    else:
        # Direct state dict
        state_dict = checkpoint
    
    # Infer hidden layer sizes from checkpoint
    hidden_layers = []
    layer_idx = 0
    while f'layers.{layer_idx}.weight' in state_dict:
        weight_shape = state_dict[f'layers.{layer_idx}.weight'].shape
        hidden_layers.append(weight_shape[0])  # Output dimension of this layer
        layer_idx += 1
    
    # Remove the last layer size (it's the action size, not a hidden layer)
    if hidden_layers:
        hidden_layers = hidden_layers[:-1]
    
    # Create network with correct architecture
    network = Network(state_size, action_size, hidden_layers=hidden_layers).to(device)
    network.load_state_dict(state_dict)
    
    network.eval()
    return network, device


def trained_agent_policy(network, device, state):
    """Get action from trained agent.
    
    Parameters
    ----------
    network : Network
        Trained neural network
    device : torch.device
        Computation device
    state : np.ndarray
        Current state
        
    Returns
    -------
    int
        Selected action
    """
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action_values = network(state_tensor)
    return action_values.cpu().data.numpy().argmax()
