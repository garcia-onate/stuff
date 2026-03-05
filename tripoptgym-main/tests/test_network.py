"""
Unit tests for neural network module.
"""

import pytest
import torch
import numpy as np
from tripoptgym.agents.network import Network


class TestNetwork:
    """Tests for Network class."""
    
    def test_initialization(self):
        """Test network can be initialized with different architectures."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        assert network is not None
        assert network.fc1.in_features == 10
        assert network.fc1.out_features == 64
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 5), f"Expected shape (1, 5), got {output.shape}"
    
    def test_forward_pass_batch(self):
        """Test forward pass with batch input."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        batch_size = 32
        states = torch.randn(batch_size, 10)
        outputs = network(states)
        
        assert outputs.shape == (batch_size, 5)
    
    def test_single_hidden_layer(self):
        """Test network with single hidden layer."""
        network = Network(state_size=10, action_size=5, hidden_layers=[128])
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 5)
    
    def test_multiple_hidden_layers(self):
        """Test network with multiple hidden layers."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 128, 64])
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 5)
    
    def test_gradient_flow(self):
        """Test that gradients flow through network."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        state = torch.randn(1, 10, requires_grad=True)
        output = network(state)
        loss = output.sum()
        loss.backward()
        
        assert state.grad is not None
        assert network.fc1.weight.grad is not None
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        network.eval()
        
        state = torch.randn(1, 10)
        output1 = network(state)
        output2 = network(state)
        
        assert torch.allclose(output1, output2)
    
    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        network.eval()
        
        state1 = torch.randn(1, 10)
        state2 = torch.randn(1, 10)
        
        output1 = network(state1)
        output2 = network(state2)
        
        # Outputs should be different (with very high probability)
        assert not torch.allclose(output1, output2)
    
    def test_numpy_input(self):
        """Test that network can handle numpy array input."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        state_np = np.random.randn(1, 10).astype(np.float32)
        state_tensor = torch.from_numpy(state_np)
        
        output = network(state_tensor)
        assert output.shape == (1, 5)


class TestNetworkParameterCount:
    """Tests for network parameter counting and initialization."""
    
    def test_parameter_count(self):
        """Test that network has correct number of parameters."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64])
        
        # First layer: 10 * 64 + 64 (bias) = 704
        # Output layer: 64 * 5 + 5 (bias) = 325
        # Total: 1029
        total_params = sum(p.numel() for p in network.parameters())
        expected_params = (10 * 64 + 64) + (64 * 5 + 5)
        
        assert total_params == expected_params
    
    def test_all_parameters_trainable(self):
        """Test that all parameters require gradients."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64, 64])
        
        for param in network.parameters():
            assert param.requires_grad


class TestNetworkDeviceHandling:
    """Tests for device handling (CPU/CUDA)."""
    
    def test_cpu_network(self):
        """Test network works on CPU."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64])
        state = torch.randn(1, 10)
        
        output = network(state)
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_network(self):
        """Test network can be moved to CUDA."""
        network = Network(state_size=10, action_size=5, hidden_layers=[64])
        network = network.cuda()
        state = torch.randn(1, 10).cuda()
        
        output = network(state)
        assert output.device.type == 'cuda'
