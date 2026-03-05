"""
Unit tests for TripOptWorldEnv environment.
"""

import pytest
import gymnasium as gym
import numpy as np
import pandas as pd
import tempfile
import os


@pytest.fixture
def sample_route_csv():
    """Create a sample route CSV file for testing."""
    data = {
        'Distance In Route': np.arange(0, 10, 0.05),
        'Effective Grade Percent': np.sin(np.arange(0, 10, 0.05)) * 2,
        'Effective Speed Limit': np.ones(200) * 60,
        'Elevation': np.arange(0, 10, 0.05) * 10
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


class TestEnvironmentInitialization:
    """Tests for environment initialization."""
    
    def test_env_creation(self, sample_route_csv):
        """Test that environment can be created."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        assert env is not None
        env.close()
    
    def test_observation_space_shape(self, sample_route_csv):
        """Test observation space has correct shape."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        
        # Should be 12-dimensional
        assert env.observation_space.shape == (12,)
        env.close()
    
    def test_action_space_size(self, sample_route_csv):
        """Test action space has correct number of actions."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        
        # Should have 5 discrete actions
        assert env.action_space.n == 5
        env.close()


class TestEnvironmentReset:
    """Tests for environment reset functionality."""
    
    def test_reset_returns_observation(self, sample_route_csv):
        """Test that reset returns valid observation."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        obs, info = env.reset()
        
        assert obs is not None
        assert obs.shape == (12,)
        assert isinstance(info, dict)
        env.close()
    
    def test_reset_observation_values(self, sample_route_csv):
        """Test that reset observation has reasonable values."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        obs, info = env.reset()
        
        # Check all values are finite
        assert np.all(np.isfinite(obs))
        
        # Position should be at start
        position_idx = 0  # First element is position
        assert obs[position_idx] >= 0
        
        env.close()
    
    def test_reset_is_deterministic_with_seed(self, sample_route_csv):
        """Test that reset with same seed produces same observation."""
        env1 = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env2 = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        assert np.allclose(obs1, obs2)
        
        env1.close()
        env2.close()


class TestEnvironmentStep:
    """Tests for environment step functionality."""
    
    def test_step_returns_correct_types(self, sample_route_csv):
        """Test that step returns correct types."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_step_observation_shape(self, sample_route_csv):
        """Test that step returns correct observation shape."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (12,)
        
        env.close()
    
    def test_all_actions_valid(self, sample_route_csv):
        """Test that all actions can be executed."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        # Try all 5 actions
        for action in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            
            if terminated or truncated:
                env.reset()
        
        env.close()
    
    def test_invalid_action_raises_error(self, sample_route_csv):
        """Test that invalid action raises appropriate error."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        # Action -1 is invalid
        with pytest.raises((AssertionError, ValueError, IndexError)):
            env.step(-1)
        
        # Action 5 is invalid (only 0-4 are valid)
        with pytest.raises((AssertionError, ValueError, IndexError)):
            env.step(5)
        
        env.close()
    
    def test_reward_is_finite(self, sample_route_csv):
        """Test that rewards are always finite."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        for _ in range(100):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            
            if terminated or truncated:
                env.reset()
        
        env.close()


class TestEnvironmentTermination:
    """Tests for environment termination conditions."""
    
    def test_episode_terminates(self, sample_route_csv):
        """Test that episode eventually terminates."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv, start_dist=0, end_dist=2)
        env.reset()
        
        max_steps = 10000
        terminated = False
        truncated = False
        
        for step in range(max_steps):
            _, _, terminated, truncated, _ = env.step(3)  # High throttle
            if terminated or truncated:
                break
        
        assert terminated or truncated, "Episode should terminate within max steps"
        env.close()
    
    def test_terminal_reward(self, sample_route_csv):
        """Test that reaching destination gives terminal reward."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv, start_dist=0, end_dist=1)
        env.reset()
        
        total_reward = 0
        terminated = False
        truncated = False
        
        # Run until termination
        for _ in range(10000):
            _, reward, terminated, truncated, _ = env.step(3)
            total_reward += reward
            if terminated or truncated:
                break
        
        # Should have accumulated positive reward
        assert total_reward > 0, "Should accumulate positive reward"
        env.close()


class TestEnvironmentObservationSpace:
    """Tests for observation space components."""
    
    def test_observation_components(self, sample_route_csv):
        """Test that observation contains expected components."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        obs, _ = env.reset()
        
        # Observation should have 12 components
        assert len(obs) == 12
        
        # All should be finite
        assert np.all(np.isfinite(obs))
        
        env.close()
    
    def test_speed_in_observation(self, sample_route_csv):
        """Test that speed is updated in observation."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        obs1, _ = env.reset()
        
        # Take action that changes speed
        obs2, _, _, _, _ = env.step(3)  # High throttle
        
        # Observations should be different
        assert not np.allclose(obs1, obs2)
        
        env.close()


class TestEnvironmentStateConsistency:
    """Tests for state consistency across steps."""
    
    def test_position_increases(self, sample_route_csv):
        """Test that train position increases over time."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        obs1, _ = env.reset()
        
        # Take several steps with throttle
        for _ in range(10):
            obs2, _, terminated, truncated, _ = env.step(3)
            if terminated or truncated:
                break
        
        # Position should have increased (assuming not immediately terminated)
        # Position is first element
        if not (terminated or truncated):
            assert obs2[0] > obs1[0], "Position should increase"
        
        env.close()
    
    def test_state_stays_in_bounds(self, sample_route_csv):
        """Test that state values stay within reasonable bounds."""
        env = gym.make('TripOptWorld-v0', csv_file=sample_route_csv)
        env.reset()
        
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            
            # All observations should be finite
            assert np.all(np.isfinite(obs))
            
            if terminated or truncated:
                env.reset()
        
        env.close()
