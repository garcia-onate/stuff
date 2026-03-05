# TripOptGym

A Gymnasium-compatible reinforcement learning environment for train trip optimization. Train a Deep Q-Network agent to control locomotive throttle and dynamic braking to minimize trip time while respecting speed limits and managing energy consumption.

## Features

- Realistic train physics simulation with trapezoidal integration
- Multi-locomotive consist model (3x ES44AC locomotives)
- Route-based scenarios with elevation profiles and speed limits
- Visual rendering with rolling map display and data grid
- DQN agent with experience replay and target network
- Configurable hyperparameters via YAML files
- Centralized device management (CUDA/CPU auto-detection)

## Development Environment

This project is developed using the following setup:

- **WSL (Windows Subsystem for Linux)**: Ubuntu 24.04 running on Windows, providing a native Linux environment for development
- **VS Code**: Primary IDE with remote development capabilities for integration with WSL
- **Python Extension**: Microsoft's Python extension for VS Code (`ms-python.python`) provides IntelliSense, debugging, linting, and Jupyter notebook support
- **Python Virtual Environment**: Isolated Python environment using `venv` to manage project dependencies

### Setting Up Your Development Environment

1. **Install WSL** (Windows users):
   ```bash
   wsl --install -d Ubuntu-24.04
   ```

2. **Install VS Code**:
   - Download VS Code from https://code.visualstudio.com/
   - Run the installer and follow the installation wizard

3. **Install the Remote-WSL extension**:
   - Open VS Code
   - Open Extensions view (Ctrl+Shift+X)
   - Search for "Remote - WSL" (by Microsoft)
   - Click Install

4. **Install the Python extension** in VS Code:
   - Open Extensions view (Ctrl+Shift+X)
   - Search for "Python" (by Microsoft)
   - Click Install

5. **Open your project in WSL**:
   - Open a WSL terminal
   - Navigate to your project directory
   - Run `code .` to open the project in VS Code

6. **Create a Python virtual environment** (in WSL terminal or VS Code integrated terminal):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/WSL
   # or
   venv\Scripts\activate     # On Windows
   ```

7. **Configure VS Code to use the virtual environment**:
   - Open Command Palette (Ctrl+Shift+P)
   - Search "Python: Select Interpreter"
   - Choose the interpreter from your `venv` directory

## Installation

Install the package in editable mode (in WSL terminal or VS Code integrated terminal with venv activated):

```bash
pip install -e .
```

For development with testing tools:

```bash
pip install -e ".[dev]"
```

Verify the installation:

```bash
# Check CLI is available
tripopt --help

## Quick Start

### Working Example

The RL agent training process has been successfully demonstrated on the included `route_data.csv` dataset for a 6-mile segment from mile marker 18 to mile marker 24. This example shows that the DQN agent can learn to effectively control the train to minimize trip time while respecting speed limits.

#### Recreating the Success

To recreate this working example, follow these two steps:

**1. Train the agent** (approximately 1000 episodes):
```bash
tripopt train --csv route_data.csv --start 18 --end 24 --episodes 1000 --checkpoint-interval 100 --target-score 5000
```

This will create periodic checkpoints (e.g., `checkpoint_ep100.pth`, `checkpoint_ep200.pth`) and logs in the `logs/` folder. When the 100-episode rolling average reaches the target score, training stops early and saves a final model as `train_checkpoint.pth`.

**2. Generate a demo video** with the trained agent:
```bash
tripopt demo --csv route_data.csv --agent trained --model train_checkpoint.pth --output demo.mp4 --start 18 --end 24
```

#### Example Output

Here's a video demonstration of the trained agent successfully navigating the route from mile 18 to mile 24:

![Trained Agent Demo](docs/trained_agent_video_final.mp4)

*The trained agent demonstrates smooth throttle control, proactive braking before speed limit reductions, and efficient trip time optimization while maintaining speed compliance.*

### Command Line Interface

TripOptGym provides a command line interface through the `tripopt` command. The CLI supports three main commands: `train` (train a DQN agent), `demo` (run demonstrations with trained or heuristic agents), and `convert` (convert route data formats). Details for each command are provided below.

#### Train Command

Train a new DQN agent:
```bash
tripopt train --csv route_data.csv --episodes 1000
```

With custom configuration:
```bash
tripopt train --config configs/default_config.yaml --csv route_data.csv --episodes 1000
```

Resume from checkpoint:
```bash
tripopt train --csv route_data.csv --resume checkpoint_ep500.pth --episodes 1000
```

With early termination (stops training when 100-episode rolling average reaches target):
```bash
tripopt train --csv route_data.csv --episodes 2000 --target-score 900
```

**Available arguments:**
- `--csv PATH` (required) - Path to route CSV file
- `--config PATH` - Path to configuration YAML file (default: `configs/default_config.yaml`)
- `--episodes INT` - Number of training episodes (default: from config or `1000`)
- `--start FLOAT` - Starting distance in miles (default: from config or `0.0`)
- `--end FLOAT` - Ending distance in miles (default: `None` = end of route)
- `--resume PATH` - Path to checkpoint file to resume from
- `--checkpoint-interval INT` - Save checkpoint every N episodes (default: from config or `100`)
- `--target-score FLOAT` - Target 100-episode average score for early termination (default: `None` = no early termination)
- `--device {auto,cuda,cpu}` - Device preference (overrides config file)

#### Demo Command

Demo with trained agent:
```bash
tripopt demo --csv route_data.csv --agent trained --model train_checkpoint.pth --output demo.mp4
```

Demo with heuristic agent:
```bash
tripopt demo --csv route_data.csv --agent heuristic --output demo_heuristic.mp4
```

**Available arguments:**
- `--csv PATH` (required) - Path to route CSV file
- `--agent {heuristic,trained}` (required) - Agent type: heuristic or trained
- `--model PATH` - Path to trained model checkpoint (required for `--agent trained`)
- `--output PATH` - Output file path (default: `demo.mp4`). Use `.csv` extension for CSV output, `.mp4` for video
- `--output-type {video,csv}` - Output type: video or csv. If not specified, auto-detects from output file extension
- `--start FLOAT` - Starting distance in miles (default: from config or `0.0`)
- `--end FLOAT` - Ending distance in miles (default: `None` = end of route)
- `--device {auto,cuda,cpu}` - Device preference

#### Convert Command

The convert command transforms route data into the standardized CSV format required by TripOptGym.

**Convert data recorder files (default)**

For converting data recorder CSV files (typically sampled at 1-second intervals) to the standardized format:

```bash
tripopt convert dr_input.csv dr_output.csv
```

Or explicitly specify the dr type:

```bash
tripopt convert --type dr dr_input.csv dr_output.csv
```

This processes high-resolution time-series data and resamples it at 0.05 mile intervals for use in training.

**Convert parsed RTC route text files**

Converts parsed RTC route text files (containing Terrain Entity Tables, Effective Grade Tables, and Speed Limit Tables) into the standardized CSV format. This is useful when you have route data exported from tripopt rtc logs.

```bash
tripopt convert --type rtc route_input_parsed.txt route_data_generated.csv
```

The input file should contain three tables:
- **Terrain Entity Table**: DIR, Sup Elev, Grade, Curve
- **EFFECTIVE_GRADE_TABLE**: Distance In Route, Effective Grade
- **Speed Limit Entity Table**: DIR, Civil Speed Limit, Effective Speed Limit

The output CSV will have these columns at 0.05 mile intervals:
- `Distance In Route` - Mile markers along the route
- `Effective Grade Percent` - Interpolated grade percentages
- `Effective Speed Limit` - Speed limits (stepwise constant between changes)
- `Elevation` - Calculated elevation in feet

**Arguments:**
- `--type {rtc,dr}` - Conversion type: `rtc` for parsed RTC route text files, `dr` for data recorder CSV files (default)
- `input_file` (required) - Path to input file (parsed route text or data recorder CSV)
- `output_file` (required) - Path to output CSV file

### Python API

The Python API provides direct access to the environment and agent components for advanced use cases such as custom training loops, integration with other frameworks, research experiments, or when you need fine-grained control over the training process beyond what the CLI commands offer.

```python
import gymnasium as gym
from tripoptgym.agents.dqn import Agent
from tripoptgym.utils.config import load_config
from tripoptgym.utils.device import get_device

# Load configuration
config = load_config('configs/default_config.yaml')

# Create environment
env = gym.make('TripOptWorld-v1', 
               route_csv_path='route_data.csv',
               start_location=0.0,
               end_location=None)

# Get device
device = get_device(config.get('device', {}).get('preference', 'auto'))

# Create agent
agent = Agent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    learning_rate=config['training']['learning_rate'],
    replay_buffer_size=config['training']['replay_buffer_size'],
    device=device,
    hidden_layers=config['network']['hidden_layers']
)

# Training loop
epsilon = config['training']['epsilon_start']
for episode in range(1000):
    state, _ = env.reset()
    done = False
    episode_score = 0
    
    while not done:
        # Select action using epsilon-greedy policy
        action = agent.act(state, epsilon)
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience and learn
        agent.step(state, action, reward, next_state, done,
                   minibatch_size=config['training']['minibatch_size'],
                   discount_factor=config['training']['discount_factor'],
                   interpolation_parameter=config['training']['interpolation_parameter'])
        
        state = next_state
        episode_score += reward
    
    # Decay epsilon
    epsilon = max(config['training']['epsilon_end'], 
                  epsilon * config['training']['epsilon_decay'])
    
    print(f"Episode {episode + 1}: Score = {episode_score:.2f}, Epsilon = {epsilon:.3f}")

env.close()
```

## Configuration

Edit [configs/default_config.yaml](configs/default_config.yaml) to customize training behavior, network architecture, and logging options.

### Training Parameters

- `learning_rate` (default: `0.0005`) - Adam optimizer learning rate. Lower values (e.g., `0.0001`) provide more stable but slower learning; higher values (e.g., `0.001`) speed up training but may be less stable.
- `minibatch_size` (default: `100`) - Number of experiences sampled from replay buffer for each learning update. Larger batches provide more stable gradients.
- `discount_factor` (default: `0.99`) - Gamma value for future reward discounting. Higher values (closer to 1.0) make the agent prioritize long-term rewards.
- `replay_buffer_size` (default: `100000`) - Maximum number of experiences stored in memory for experience replay.
- `interpolation_parameter` (default: `0.001`) - Tau for soft target network updates. Controls how quickly the target network tracks the local network.
- `update_frequency` (default: `4`) - How often the agent learns from experiences (every N steps). Lower values (e.g., `1`) update every step for more responsive learning but slower training; higher values (e.g., `8`, `16`) reduce computational overhead but may miss learning opportunities.
- `epsilon_start` (default: `1.0`) - Initial exploration rate. Start with full exploration (1.0) and decay over time.
- `epsilon_end` (default: `0.01`) - Minimum exploration rate to maintain some randomness throughout training.
- `epsilon_decay` (default: `0.995`) - Multiplicative decay rate per episode. With this value, epsilon decays from 1.0 to 0.01 over approximately 1000 episodes.
- `number_episodes` (default: `1000`) - Default number of training episodes if not specified via command line.
- `max_timesteps` (default: `25000`) - Maximum steps per episode before truncation.
- `checkpoint_interval` (default: `100`) - Save checkpoint every N episodes for backup and resumption.

### Network Architecture

- `hidden_layers` (default: `[64, 64]`) - List of hidden layer sizes for the Q-network. Try `[128, 128]` for more capacity or `[32, 32]` for faster training on smaller problems.

### Environment Settings

- `num_parallel_envs` (default: `4`) - Number of environments to run in parallel for faster data collection. Set to `1` for sequential execution.
- `start_location` (default: `18`) - Default starting mile marker for route if not overridden by command line.
- `end_location` (default: `24`) - Default ending mile marker for route if not overridden by command line.

### Device Configuration

- `preference` (default: `auto`) - Device for training: `auto` (automatically detect CUDA), `cuda` (force GPU), or `cpu` (force CPU).

### Logging Configuration

- `enable_logging` (default: `true`) - Enable comprehensive training metrics logging to files and TensorBoard.
- `log_dir` (default: `logs`) - Directory where log files and TensorBoard events are saved.
- `enable_tensorboard` (default: `true`) - Enable TensorBoard visualization. Requires tensorboard package.
- `log_step_level` (default: `true`) - Log detailed metrics at each learning step. Provides granular data but creates larger log files.
- `run_name` (default: `null`) - Optional custom name for the training run. If `null`, uses timestamp.

## Project Structure

```
tripoptgym/
├── environment/
│   ├── physics.py       # Train dynamics and locomotive model
│   └── env.py           # Gymnasium environment
├── agents/
│   ├── network.py       # Neural network architecture
│   ├── dqn.py          # DQN agent with experience replay
│   ├── heuristic.py    # Rule-based policy
│   └── inference.py    # Model loading utilities
├── visualization/
│   ├── rendering.py    # RollingMap and DataGridView
│   ├── video.py        # Video generation
│   └── sim_to_csv.py   # CSV output generation
├── utils/
│   ├── device.py           # Centralized CUDA/CPU management
│   ├── config.py           # YAML configuration loading
│   ├── route_converter.py  # Route data conversion
│   ├── training_logger.py  # Training metrics and TensorBoard logging
│   └── process_dr_data.py  # Data processing utilities
└── scripts/
    └── main.py         # CLI driver
```

## Observation Space

6-dimensional continuous vector:
- Train velocity (mph)
- Train acceleration (mph/minute)
- Current position in route (miles)
- Current speed limit (mph)
- Next speed limit (mph)
- Next speed limit location (miles)

## Action Space

3 discrete actions (notch-based control):
- 0: Hold current notch (no change)
- 1: Notch up (increase throttle or decrease brake)
- 2: Notch down (decrease throttle or increase brake)

Notch ranges from -8 (maximum dynamic brake) to +8 (maximum throttle). Actions are rate-limited to one notch change per 3 seconds.

## Reward Structure

Multi-component reward function:
1. **Progress reward**: +100 points per mile traveled
2. **Speed compliance**: 
   - Quadratic penalty for exceeding speed limits (-5 × error^1.5 for minor violations, up to -50 per mph for major violations)
   - Bonus for operating near speed limit (85-100% of limit)
   - Modest penalty for very slow speeds (<40% of limit) without acceleration
3. **Anticipation bonus**: Up to +10 points for slowing down early before speed limit reductions (within 2 miles)
4. **Terminal rewards**: 
   - +500 for successfully reaching destination
   - -300 for stalling (speed drops below 5 mph)

# Training Monitoring Guide

This guide explains how to use the comprehensive training monitoring system for TripOptGym DQN training.

## Overview

The monitoring system provides three complementary outputs:
1. **Console Output**: Real-time training progress
2. **CSV Logs**: Complete metric history for analysis
3. **TensorBoard**: Interactive visualization dashboards

## Quick Start

### Enable Logging (Default)

Logging is enabled by default. Simply run training:

```bash
tripopt train --config configs/default_config.yaml --csv route_data.csv
```

This will create a `logs/` directory with timestamped run folders containing:
- `episode_metrics.csv` - Episode-level metrics
- `step_metrics.csv` - Step-level metrics (if enabled)
- `tensorboard/` - TensorBoard event files

### View TensorBoard

Launch TensorBoard to view real-time plots:

```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006`

## Configuration Options

Edit `configs/default_config.yaml` to customize logging:

```yaml
logging:
  enable_logging: true           # Enable/disable logging
  log_dir: logs                  # Directory for log files
  enable_tensorboard: true       # Enable TensorBoard
  log_step_level: false          # Log every learning step (detailed)
  run_name: null                 # Custom run name (null = timestamp)
```

### Step-Level vs Episode-Level Logging

- **Episode-level** (default): Logs aggregate metrics per episode. Recommended for most use cases.
- **Step-level**: Logs metrics every time the agent learns (every 4 steps). More detailed but creates larger files. Useful for debugging training instability.

To enable step-level logging:

```yaml
logging:
  log_step_level: true
```

## Tracked Metrics

### Episode-Level Metrics (CSV columns)

| Metric | Description |
|--------|-------------|
| `episode` | Episode number |
| `timestep` | Global timestep counter |
| `score` | Total episode reward |
| `steps` | Steps in episode |
| `epsilon` | Current exploration rate |
| `avg_score_100` | Rolling 100-episode average score |
| `loss_mean` | Mean TD loss for episode |
| `loss_std` | Std dev of TD loss |
| `q_expected_mean` | Mean Q-values predicted |
| `q_target_mean` | Mean target Q-values |
| `td_error_mean` | Mean temporal difference error |
| `gradient_norm` | L2 norm of gradients |
| `param_norm` | L2 norm of network parameters |
| `reward_progress` | Progress reward component |
| `reward_speed_compliance` | Speed compliance reward |
| `reward_anticipation` | Anticipation reward |
| `reward_terminal` | Terminal reward |
| `termination_reason` | 1=Destination, 2=Stalled, 3=Overspeed |
| `success_rate_100` | Rolling 100-episode success rate |
| `action_0_count` | Count of action 0 taken |
| `action_1_count` | Count of action 1 taken |
| `action_2_count` | Count of action 2 taken |
| `buffer_size` | Replay buffer utilization |
| `wall_time` | Training time elapsed (seconds) |

### Step-Level Metrics (when enabled)

| Metric | Description |
|--------|-------------|
| `episode` | Episode number |
| `step` | Step within episode |
| `timestep` | Global timestep |
| `loss` | TD loss for this learning step |
| `q_expected` | Expected Q-value |
| `q_target` | Target Q-value |
| `td_error` | Temporal difference error |
| `gradient_norm` | Gradient norm |
| `param_norm` | Parameter norm |
| `buffer_size` | Buffer size |

## TensorBoard Visualizations

TensorBoard organizes metrics into tabs:

### Episode Metrics
- Score per episode
- 100-episode rolling average
- Success rate
- Episode length

### Learning Metrics
- Loss (mean, std)
- Q-values (expected vs target)
- TD error
- Gradient norm
- Parameter norm

### Rewards
- Progress reward
- Speed compliance reward
- Anticipation reward
- Terminal reward

### Actions
- Action distribution (ratio of each action)

### Memory
- Replay buffer utilization

### Step-Level (if enabled)
- Loss per learning step
- Q-values per step
- TD error per step

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=tripoptgym --cov-report=html tests/
```

## TODO

The following enhancements are planned for future releases:

1. **Configurable Train Characteristics**: Replace hard-coded train parameters (Davis coefficients, locomotive model, train mass) with configurable options. Currently these are fixed in the environment; should support custom train consists, different locomotive types, and varying train weights through config files or environment parameters.

2. **In-Train Forces Integration**: Incorporate [pysim](https://gitlab.corp.wabtec.com/Joseph.Wakeman/pysim) to enable in-train force calculations as part of the simulation environment. This would provide more realistic modeling of coupler forces, slack action, and longitudinal train dynamics.

