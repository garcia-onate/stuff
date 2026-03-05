# Quick Start Guide

Get up and running with TripOptGym in 5 minutes.

## Installation

```bash
cd /home/jwakeman/tripoptgym
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check CLI is available
tripopt --help

# Run tests
pytest tests/ -v
```

## Train Your First Agent

```bash
# Train for 100 episodes (takes ~5-10 minutes)
tripopt train \
  --csv route_data.csv \
  --episodes 100 \
  --checkpoint-interval 25

# This creates: checkpoint_ep25.pth, checkpoint_ep50.pth, checkpoint_ep75.pth, checkpoint_ep100.pth
```

## Create a Demo Video

### Option 1: Heuristic Agent
```bash
tripopt demo \
  --csv route_data.csv \
  --agent heuristic \
  --output demo_heuristic.mp4
```

### Option 2: Trained Agent
```bash
tripopt demo \
  --csv route_data.csv \
  --agent trained \
  --model checkpoint_ep100.pth \
  --output demo_trained.mp4
```

## Python API Usage

### Training Script

```python
import gymnasium as gym
from tripoptgym.agents.dqn import Agent
from tripoptgym.utils.config import load_config

# Load configuration
config = load_config('configs/default_config.yaml')

# Create environment
env = gym.make('TripOptWorld-v0', csv_file='route_data.csv')

# Create agent
agent = Agent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    config=config
)

# Training loop
for episode in range(100):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.epsilon_greedy_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

# Save final model
agent.save_checkpoint('my_checkpoint.pth')
env.close()
```

### Inference Script

```python
import gymnasium as gym
from tripoptgym.agents.inference import load_trained_agent, trained_agent_policy
from tripoptgym.utils.device import get_device

# Setup
device = get_device()
env = gym.make('TripOptWorld-v0', csv_file='route_data.csv')

# Load trained model
model = load_trained_agent(
    'my_checkpoint.pth',
    state_size=12,
    action_size=5,
    device=device
)

# Run episode
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = trained_agent_policy(state, model, device)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Episode reward: {total_reward:.2f}")
env.close()
```

## Customize Configuration

Edit `configs/default_config.yaml`:

```yaml
training:
  learning_rate: 0.0005      # Lower for stability, higher for speed
  batch_size: 64             # Larger for stability
  gamma: 0.99                # Discount factor
  epsilon_start: 1.0         # Start with full exploration
  epsilon_min: 0.01          # Minimum exploration
  epsilon_decay: 0.995       # Decay rate per episode

network:
  hidden_layers: [128, 128]  # Try [64, 64] for faster, [256, 256] for more capacity

device:
  preference: auto           # Or 'cuda' / 'cpu'
```

Then use it:
```bash
tripopt train --config configs/default_config.yaml --csv route_data.csv
```

## Convert Route Data

If you have parsed route text files:

```bash
tripopt convert route_input_parsed.txt route_data_new.csv
```

## Common Tasks

### Resume Training from Checkpoint
```bash
tripopt train \
  --csv route_data.csv \
  --resume checkpoint_ep100.pth \
  --episodes 200
```

### Train on Custom Route Section
```bash
tripopt train \
  --csv route_data.csv \
  --start 5.0 \
  --end 15.0 \
  --episodes 100
```

### Force CPU/GPU Usage
```bash
tripopt train --csv route_data.csv --device cuda  # Force GPU
tripopt train --csv route_data.csv --device cpu   # Force CPU
```

### Compare Heuristic vs Trained Agent
```bash
# Generate both videos
tripopt demo --csv route_data.csv --agent heuristic --output heuristic.mp4
tripopt demo --csv route_data.csv --agent trained --model checkpoint.pth --output trained.mp4

# Compare side-by-side in video player
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_physics.py

# With coverage
pytest --cov=tripoptgym --cov-report=html tests/
# Open htmlcov/index.html

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

## Troubleshooting

### "Import tripoptgym could not be resolved"
Run: `pip install -e .` from the project root

### "CUDA out of memory"
Add `--device cpu` to command or edit config:
```yaml
device:
  preference: cpu
```

### Old checkpoint won't load
See `docs/MIGRATION.md` for conversion instructions

### "No module named 'yaml'"
Install dependencies: `pip install -e .`

### Tests fail with pytest not found
Install dev dependencies: `pip install -e ".[dev]"`

## Next Steps

- Read [README.md](../README.md) for detailed documentation
- Check [MIGRATION.md](MIGRATION.md) for converting old checkpoints
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
- Explore the code in `tripoptgym/` to understand the implementation

## Getting Help

1. Check the documentation in `docs/`
2. Read docstrings in the code
3. Look at test files in `tests/` for usage examples
4. Review the original scripts for algorithm details

Happy training! 🚂
