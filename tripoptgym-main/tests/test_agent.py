"""
Unit tests for DQN agent module.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from tripoptgym.agents.dqn import Agent, ReplayMemory


class TestReplayMemory:
    """Tests for ReplayMemory class."""
    
    def test_initialization(self):
        """Test replay memory can be initialized."""
        memory = ReplayMemory(capacity=1000)
        assert len(memory) == 0
        assert memory.capacity == 1000
    
    def test_push_and_length(self):
        """Test pushing transitions and checking length."""
        memory = ReplayMemory(capacity=100)
        
        for i in range(50):
            memory.push(
                state=np.array([i]),
                action=i % 5,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False
            )
        
        assert len(memory) == 50
    
    def test_circular_buffer(self):
        """Test that memory acts as circular buffer when full."""
        capacity = 10
        memory = ReplayMemory(capacity=capacity)
        
        # Fill beyond capacity
        for i in range(20):
            memory.push(
                state=np.array([i]),
                action=i % 5,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False
            )
        
        # Length should be capped at capacity
        assert len(memory) == capacity
    
    def test_sample(self):
        """Test sampling from memory."""
        memory = ReplayMemory(capacity=100)
        
        # Add transitions
        for i in range(50):
            memory.push(
                state=np.array([i, i*2]),
                action=i % 5,
                reward=float(i),
                next_state=np.array([i+1, (i+1)*2]),
                done=(i % 10 == 0)
            )
        
        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        assert states.shape == (batch_size, 2)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 2)
        assert dones.shape == (batch_size,)
    
    def test_cannot_sample_more_than_available(self):
        """Test that sampling fails when not enough transitions."""
        memory = ReplayMemory(capacity=100)
        
        # Add only 10 transitions
        for i in range(10):
            memory.push(
                state=np.array([i]),
                action=i % 5,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False
            )
        
        # Should not be able to sample 32
        with pytest.raises((ValueError, IndexError)):
            memory.sample(32)


class TestAgent:
    """Tests for Agent class."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = {
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'gamma': 0.99,
                'tau': 0.001,
                'epsilon_start': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 10000
            },
            'network': {
                'hidden_layers': [64, 64]
            }
        }
    
    def test_initialization(self):
        """Test agent can be initialized."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        assert agent is not None
        assert agent.state_size == 12
        assert agent.action_size == 5
        assert agent.epsilon == 1.0
    
    def test_epsilon_greedy_action_exploration(self):
        """Test epsilon-greedy action selection during exploration."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        agent.epsilon = 1.0  # Force exploration
        
        state = np.random.randn(12)
        
        # With epsilon=1.0, should always return random action
        actions = [agent.epsilon_greedy_action(state) for _ in range(100)]
        
        # Should see variety of actions
        unique_actions = len(set(actions))
        assert unique_actions > 1, "Should explore multiple actions"
    
    def test_epsilon_greedy_action_exploitation(self):
        """Test epsilon-greedy action selection during exploitation."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        agent.epsilon = 0.0  # Force exploitation
        
        state = np.random.randn(12)
        
        # With epsilon=0.0, should always return same action for same state
        actions = [agent.epsilon_greedy_action(state) for _ in range(10)]
        assert len(set(actions)) == 1, "Should exploit same action"
    
    def test_epsilon_decay(self):
        """Test epsilon decays over time."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        initial_epsilon = agent.epsilon
        
        # Perform learning step
        state = np.random.randn(12)
        next_state = np.random.randn(12)
        agent.learn(state, action=0, reward=1.0, next_state=next_state, done=False)
        
        # After learning, if memory is too small, epsilon won't decay yet
        # So let's fill memory first
        for _ in range(100):
            agent.learn(state, action=0, reward=1.0, next_state=next_state, done=False)
        
        # Now epsilon should have decayed
        assert agent.epsilon <= initial_epsilon
    
    def test_action_bounds(self):
        """Test that actions are within valid range."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        state = np.random.randn(12)
        
        for _ in range(100):
            action = agent.epsilon_greedy_action(state)
            assert 0 <= action < 5, f"Action {action} out of bounds"
    
    def test_greedy_action(self):
        """Test greedy action selection."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        state = np.random.randn(12)
        
        action = agent.greedy_action(state)
        assert 0 <= action < 5
    
    def test_learn_without_enough_memory(self):
        """Test learning doesn't crash when memory is insufficient."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        state = np.random.randn(12)
        next_state = np.random.randn(12)
        
        # Should not crash even with insufficient memory
        agent.learn(state, action=0, reward=1.0, next_state=next_state, done=False)
    
    def test_learn_with_sufficient_memory(self):
        """Test learning proceeds when memory is sufficient."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        state = np.random.randn(12)
        next_state = np.random.randn(12)
        
        # Fill memory
        for i in range(100):
            agent.learn(
                state=state + i * 0.1,
                action=i % 5,
                reward=float(i),
                next_state=next_state + i * 0.1,
                done=(i % 20 == 0)
            )
        
        assert len(agent.memory) == 100
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading agent checkpoints."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        
        # Train a bit to change network weights
        state = np.random.randn(12)
        next_state = np.random.randn(12)
        for _ in range(50):
            agent.learn(state, action=0, reward=1.0, next_state=next_state, done=False)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pth")
            agent.save_checkpoint(checkpoint_path)
            
            # Create new agent and load checkpoint
            new_agent = Agent(state_size=12, action_size=5, config=self.config)
            new_agent.load_checkpoint(checkpoint_path)
            
            # Compare outputs
            test_state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            old_output = agent.local_network(test_state)
            new_output = new_agent.local_network(test_state)
            
            assert torch.allclose(old_output, new_output, rtol=1e-5)


class TestAgentIntegration:
    """Integration tests for Agent with environment."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = {
            'training': {
                'learning_rate': 0.001,
                'batch_size': 8,  # Small batch for testing
                'gamma': 0.99,
                'tau': 0.001,
                'epsilon_start': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 1000
            },
            'network': {
                'hidden_layers': [32, 32]  # Smaller network for testing
            }
        }
    
    def test_training_loop(self):
        """Test a simple training loop."""
        agent = Agent(state_size=12, action_size=5, config=self.config)
        
        # Simulate training episodes
        num_steps = 100
        for step in range(num_steps):
            state = np.random.randn(12)
            action = agent.epsilon_greedy_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = (step % 20 == 0)
            
            agent.learn(state, action, reward, next_state, done)
        
        # Check that memory has been filled
        assert len(agent.memory) == num_steps
        
        # Check that epsilon has decayed
        assert agent.epsilon < 1.0
