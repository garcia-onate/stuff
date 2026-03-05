"""Neural network architecture for DQN agent.

This module contains the neural network model used by the Deep Q-Learning agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """Neural network for DQN agent.
    
    A fully-connected neural network with configurable hidden layers
    for value function approximation in Deep Q-Learning.
    
    Parameters
    ----------
    state_size : int
        Dimension of state space
    action_size : int
        Number of possible actions
    hidden_layers : list of int, optional
        Sizes of hidden layers (default: [64, 64])
    seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Examples
    --------
    >>> network = Network(state_size=6, action_size=3)
    >>> network = Network(6, 3, hidden_layers=[128, 128, 64])
    """
    
    def __init__(self, state_size, action_size, hidden_layers=None, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        if hidden_layers is None:
            hidden_layers = [64, 64]
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], action_size))

    def forward(self, state):
        """Forward pass through the network.
        
        Parameters
        ----------
        state : torch.Tensor
            Input state tensor
            
        Returns
        -------
        torch.Tensor
            Action values (Q-values)
        """
        x = state
        # Apply ReLU to all but the last layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # No activation on output layer
        return self.layers[-1](x)
