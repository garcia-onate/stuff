"""Deep Q-Learning agent implementation.

This module contains the DQN agent and experience replay components.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from tripoptgym.agents.network import Network
from tripoptgym.utils.device import get_device


class ReplayMemory:
    """Experience replay buffer for DQN training.
    
    Parameters
    ----------
    capacity : int
        Maximum number of experiences to store
    device : torch.device, optional
        Device for tensor operations
    """
    
    def __init__(self, capacity, device=None):
        self.device = device if device is not None else get_device()
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        """Store an experience tuple.
        
        Parameters
        ----------
        event : tuple
            (state, action, reward, next_state, done)
        """
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """Sample a batch of experiences.
        
        Parameters
        ----------
        batch_size : int
            Number of experiences to sample
            
        Returns
        -------
        tuple
            (states, next_states, actions, rewards, dones) as tensors
        """
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones


class Agent:
    """Deep Q-Network agent.
    
    Implements DQN with experience replay and target network for stable training.
    
    Parameters
    ----------
    state_size : int
        Dimension of state space
    action_size : int
        Number of possible actions
    learning_rate : float, optional
        Learning rate for optimizer (default: 5e-4)
    replay_buffer_size : int, optional
        Size of experience replay buffer (default: 100000)
    device : torch.device, optional
        Device for computation
    hidden_layers : list of int, optional
        Hidden layer sizes for network
    update_frequency : int, optional
        Learn every N steps (default: 4)
    """
    
    def __init__(self, state_size, action_size, learning_rate=5e-4, 
                 replay_buffer_size=int(1e5), device=None, hidden_layers=None, update_frequency=4):
        self.device = device if device is not None else get_device()
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_qnetwork = Network(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size, device=self.device)
        self.t_step = 0
        self.update_frequency = update_frequency
        # Epsilon tracking for epsilon-greedy policy
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def step(self, state, action, reward, next_state, done, minibatch_size=100, discount_factor=0.99, interpolation_parameter=1e-3):
        """Store experience and learn if enough samples available.

        Parameters
        ----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether episode terminated
        minibatch_size : int, optional
            Batch size for learning
        discount_factor : float, optional
            Discount factor gamma
        interpolation_parameter : float, optional
            Soft update parameter tau (default: 1e-3)

        Returns
        -------
        dict or None
            Learning metrics if learning occurred, None otherwise
        """
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                return self.learn(experiences, discount_factor, interpolation_parameter)
        return None

    def act(self, state, epsilon=0.):
        """Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        epsilon : float, optional
            Exploration rate (default: 0)
            
        Returns
        -------
        int
            Selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor, interpolation_parameter=1e-3):
        """Update value parameters using given batch of experiences.

        Parameters
        ----------
        experiences : tuple
            Batch of (states, next_states, actions, rewards, dones)
        discount_factor : float
            Discount factor gamma
        interpolation_parameter : float, optional
            Soft update parameter tau

        Returns
        -------
        dict
            Learning metrics including loss, Q-values, TD error, and gradient norm
        """
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in self.local_qnetwork.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** 0.5

        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

        # Compute parameter norm
        param_norm = sum(p.data.norm(2).item() ** 2 for p in self.local_qnetwork.parameters()) ** 0.5

        # Compute TD error
        td_error = torch.abs(q_expected - q_targets).mean().item()

        # Return metrics
        return {
            'loss': loss.item(),
            'q_expected_mean': q_expected.mean().item(),
            'q_expected_std': q_expected.std().item(),
            'q_target_mean': q_targets.mean().item(),
            'q_target_std': q_targets.std().item(),
            'td_error': td_error,
            'gradient_norm': gradient_norm,
            'param_norm': param_norm
        }

    def soft_update(self, local_model, target_model, interpolation_parameter):
        """Soft update target network parameters.
        
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Parameters
        ----------
        local_model : torch.nn.Module
            Local network
        target_model : torch.nn.Module
            Target network
        interpolation_parameter : float
            Interpolation parameter τ
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

    def save_checkpoint(self, filepath, episode, epsilon, scores_on_100_episodes):
        """Save training checkpoint.
        
        Parameters
        ----------
        filepath : str
            Path to save checkpoint
        episode : int
            Current episode number
        epsilon : float
            Current epsilon value
        scores_on_100_episodes : deque
            Recent episode scores
        """
        checkpoint = {
            'episode': episode,
            'local_qnetwork_state_dict': self.local_qnetwork.state_dict(),
            'target_qnetwork_state_dict': self.target_qnetwork.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': epsilon,
            'scores_on_100_episodes': list(scores_on_100_episodes)
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint.
        
        Parameters
        ----------
        filepath : str
            Path to checkpoint file
            
        Returns
        -------
        tuple
            (episode, epsilon, scores_on_100_episodes)
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.local_qnetwork.load_state_dict(checkpoint['local_qnetwork_state_dict'])
        self.target_qnetwork.load_state_dict(checkpoint['target_qnetwork_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['epsilon'], deque(checkpoint['scores_on_100_episodes'], maxlen=100)
