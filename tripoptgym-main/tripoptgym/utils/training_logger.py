"""Training monitoring and logging utilities.

Provides comprehensive logging for DQN training including:
- CSV logging for metric history
- TensorBoard integration for real-time visualization
- Console output formatting
- Configurable granularity (episode-level vs step-level)
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TrainingLogger:
    """Comprehensive training logger with CSV and TensorBoard support.

    Parameters
    ----------
    log_dir : str
        Directory to save logs (default: 'logs')
    run_name : str, optional
        Name for this training run (default: timestamp)
    enable_csv : bool
        Whether to enable CSV file logging (default: True)
    enable_tensorboard : bool
        Whether to enable TensorBoard logging (default: True if available)
    log_step_level : bool
        Whether to log metrics at step level (default: False, episode-level only)
    """

    def __init__(
        self,
        log_dir: str = 'logs',
        run_name: Optional[str] = None,
        enable_csv: bool = True,
        enable_tensorboard: bool = True,
        log_step_level: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate run name if not provided
        if run_name is None:
            run_name = time.strftime('%Y%m%d_%H%M%S')
        self.run_name = run_name

        self.enable_csv = enable_csv
        self.log_step_level = log_step_level

        # Create run-specific directory
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging setup
        self.episode_csv_path = self.run_dir / 'episode_metrics.csv' if enable_csv else None
        self.episode_csv_file = None
        self.episode_csv_writer = None
        self.episode_fieldnames = [
            'episode', 'timestep', 'score', 'steps', 'epsilon',
            'avg_score_100', 'loss_mean', 'loss_std', 'loss_min', 'loss_max',
            'q_expected_mean', 'q_expected_std', 'q_target_mean', 'q_target_std',
            'td_error_mean', 'td_error_std',
            'reward_progress', 'reward_speed_compliance', 'reward_anticipation', 'reward_terminal',
            'termination_reason', 'success_rate_100',
            'action_0_count', 'action_1_count', 'action_2_count',
            'gradient_norm', 'param_norm',
            'buffer_size', 'wall_time'
        ]

        if log_step_level:
            self.step_csv_path = self.run_dir / 'step_metrics.csv' if enable_csv else None
            self.step_csv_file = None
            self.step_csv_writer = None
            self.step_fieldnames = [
                'episode', 'step', 'timestep', 'loss', 'q_expected', 'q_target',
                'td_error', 'gradient_norm', 'param_norm', 'buffer_size'
            ]

        # TensorBoard setup
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.enable_tensorboard:
            tb_dir = self.run_dir / 'tensorboard'
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard logging enabled: {tb_dir}")
            print(f"  View with: tensorboard --logdir {self.log_dir}")
        elif enable_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not available. Install with: pip install tensorboard")

        # Episode tracking
        self.episode_scores = []
        self.episode_successes = []  # Track if episode reached destination

        # Step-level accumulators (reset each episode)
        self.current_episode_losses = []
        self.current_episode_q_expected = []
        self.current_episode_q_targets = []
        self.current_episode_td_errors = []
        self.current_episode_gradient_norms = []
        self.current_episode_param_norms = []
        self.current_episode_actions = defaultdict(int)

        # Timing
        self.start_time = time.time()
        self.episode_start_time = time.time()

        # Global step counter
        self.global_step = 0

        if self.enable_csv:
            self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Episode-level CSV
        self.episode_csv_file = open(self.episode_csv_path, 'w', newline='')
        self.episode_csv_writer = csv.DictWriter(
            self.episode_csv_file,
            fieldnames=self.episode_fieldnames
        )
        self.episode_csv_writer.writeheader()
        self.episode_csv_file.flush()

        # Step-level CSV
        if self.log_step_level:
            self.step_csv_file = open(self.step_csv_path, 'w', newline='')
            self.step_csv_writer = csv.DictWriter(
                self.step_csv_file,
                fieldnames=self.step_fieldnames
            )
            self.step_csv_writer.writeheader()
            self.step_csv_file.flush()

    def log_step(
        self,
        episode: int,
        step: int,
        action: int,
        loss: Optional[float] = None,
        q_expected: Optional[float] = None,
        q_target: Optional[float] = None,
        td_error: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        param_norm: Optional[float] = None,
        buffer_size: Optional[int] = None
    ):
        """Log step-level metrics (when learning occurs).

        Parameters
        ----------
        episode : int
            Current episode number
        step : int
            Step within episode
        action : int
            Action taken
        loss : float, optional
            TD loss from learning step
        q_expected : float, optional
            Expected Q-value
        q_target : float, optional
            Target Q-value
        td_error : float, optional
            Temporal difference error
        gradient_norm : float, optional
            Gradient norm
        param_norm : float, optional
            Parameter norm
        buffer_size : int, optional
            Current replay buffer size
        """
        self.global_step += 1

        # Accumulate for episode summary
        self.current_episode_actions[action] += 1

        if loss is not None:
            self.current_episode_losses.append(loss)
        if q_expected is not None:
            self.current_episode_q_expected.append(q_expected)
        if q_target is not None:
            self.current_episode_q_targets.append(q_target)
        if td_error is not None:
            self.current_episode_td_errors.append(td_error)
        if gradient_norm is not None:
            self.current_episode_gradient_norms.append(gradient_norm)
        if param_norm is not None:
            self.current_episode_param_norms.append(param_norm)

        # Log to step-level CSV if enabled and learning occurred
        if self.log_step_level and loss is not None and self.enable_csv:
            step_data = {
                'episode': episode,
                'step': step,
                'timestep': self.global_step,
                'loss': loss,
                'q_expected': q_expected,
                'q_target': q_target,
                'td_error': td_error,
                'gradient_norm': gradient_norm,
                'param_norm': param_norm,
                'buffer_size': buffer_size
            }
            self.step_csv_writer.writerow(step_data)
            self.step_csv_file.flush()

            # Log to TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar('Step/Loss', loss, self.global_step)
                if q_expected is not None:
                    self.tb_writer.add_scalar('Step/Q_Expected', q_expected, self.global_step)
                if q_target is not None:
                    self.tb_writer.add_scalar('Step/Q_Target', q_target, self.global_step)
                if td_error is not None:
                    self.tb_writer.add_scalar('Step/TD_Error', td_error, self.global_step)
                if gradient_norm is not None:
                    self.tb_writer.add_scalar('Step/Gradient_Norm', gradient_norm, self.global_step)

    def log_episode(
        self,
        episode: int,
        score: float,
        steps: int,
        epsilon: float,
        reward_components: Optional[Dict[str, float]] = None,
        termination_reason: int = 0,
        buffer_size: Optional[int] = None
    ):
        """Log episode-level metrics.

        Parameters
        ----------
        episode : int
            Episode number
        score : float
            Total episode reward
        steps : int
            Number of steps in episode
        epsilon : float
            Current epsilon value
        reward_components : dict, optional
            Dictionary with keys: 'progress', 'speed_compliance', 'anticipation', 'terminal'
        termination_reason : int
            0=Unknown, 1=Destination (success), 2=Stalled, 3=Overspeed
        buffer_size : int, optional
            Current replay buffer size
        """
        # Track scores and successes
        self.episode_scores.append(score)
        self.episode_successes.append(1 if termination_reason == 1 else 0)

        # Compute rolling averages
        avg_score_100 = np.mean(self.episode_scores[-100:]) if len(self.episode_scores) >= 100 else np.mean(self.episode_scores)
        success_rate_100 = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)

        # Compute loss statistics from accumulated step data
        loss_mean = np.mean(self.current_episode_losses) if self.current_episode_losses else None
        loss_std = np.std(self.current_episode_losses) if self.current_episode_losses else None
        loss_min = np.min(self.current_episode_losses) if self.current_episode_losses else None
        loss_max = np.max(self.current_episode_losses) if self.current_episode_losses else None

        q_expected_mean = np.mean(self.current_episode_q_expected) if self.current_episode_q_expected else None
        q_expected_std = np.std(self.current_episode_q_expected) if self.current_episode_q_expected else None
        q_target_mean = np.mean(self.current_episode_q_targets) if self.current_episode_q_targets else None
        q_target_std = np.std(self.current_episode_q_targets) if self.current_episode_q_targets else None

        td_error_mean = np.mean(self.current_episode_td_errors) if self.current_episode_td_errors else None
        td_error_std = np.std(self.current_episode_td_errors) if self.current_episode_td_errors else None

        gradient_norm = np.mean(self.current_episode_gradient_norms) if self.current_episode_gradient_norms else None
        param_norm = np.mean(self.current_episode_param_norms) if self.current_episode_param_norms else None

        # Wall time
        wall_time = time.time() - self.start_time

        # Prepare episode data
        episode_data = {
            'episode': episode,
            'timestep': self.global_step,
            'score': score,
            'steps': steps,
            'epsilon': epsilon,
            'avg_score_100': avg_score_100,
            'loss_mean': loss_mean,
            'loss_std': loss_std,
            'loss_min': loss_min,
            'loss_max': loss_max,
            'q_expected_mean': q_expected_mean,
            'q_expected_std': q_expected_std,
            'q_target_mean': q_target_mean,
            'q_target_std': q_target_std,
            'td_error_mean': td_error_mean,
            'td_error_std': td_error_std,
            'termination_reason': termination_reason,
            'success_rate_100': success_rate_100,
            'action_0_count': self.current_episode_actions.get(0, 0),
            'action_1_count': self.current_episode_actions.get(1, 0),
            'action_2_count': self.current_episode_actions.get(2, 0),
            'gradient_norm': gradient_norm,
            'param_norm': param_norm,
            'buffer_size': buffer_size,
            'wall_time': wall_time
        }

        # Add reward components if provided
        if reward_components:
            episode_data['reward_progress'] = reward_components.get('progress', 0)
            episode_data['reward_speed_compliance'] = reward_components.get('speed_compliance', 0)
            episode_data['reward_anticipation'] = reward_components.get('anticipation', 0)
            episode_data['reward_terminal'] = reward_components.get('terminal', 0)

        # Write to CSV
        if self.enable_csv:
            self.episode_csv_writer.writerow(episode_data)
            self.episode_csv_file.flush()

        # Log to TensorBoard
        if self.tb_writer:
            # Episode metrics
            self.tb_writer.add_scalar('Episode/Score', score, episode)
            self.tb_writer.add_scalar('Episode/Avg_Score_100', avg_score_100, episode)
            self.tb_writer.add_scalar('Episode/Steps', steps, episode)
            self.tb_writer.add_scalar('Episode/Epsilon', epsilon, episode)
            self.tb_writer.add_scalar('Episode/Success_Rate_100', success_rate_100, episode)

            # Loss and Q-value metrics
            if loss_mean is not None:
                self.tb_writer.add_scalar('Learning/Loss_Mean', loss_mean, episode)
                self.tb_writer.add_scalar('Learning/Loss_Std', loss_std, episode)
            if q_expected_mean is not None:
                self.tb_writer.add_scalar('Learning/Q_Expected_Mean', q_expected_mean, episode)
            if q_target_mean is not None:
                self.tb_writer.add_scalar('Learning/Q_Target_Mean', q_target_mean, episode)
            if td_error_mean is not None:
                self.tb_writer.add_scalar('Learning/TD_Error_Mean', td_error_mean, episode)
            if gradient_norm is not None:
                self.tb_writer.add_scalar('Learning/Gradient_Norm', gradient_norm, episode)
            if param_norm is not None:
                self.tb_writer.add_scalar('Learning/Param_Norm', param_norm, episode)

            # Reward components
            if reward_components:
                self.tb_writer.add_scalar('Rewards/Progress', reward_components['progress'], episode)
                self.tb_writer.add_scalar('Rewards/Speed_Compliance', reward_components['speed_compliance'], episode)
                self.tb_writer.add_scalar('Rewards/Anticipation', reward_components['anticipation'], episode)
                self.tb_writer.add_scalar('Rewards/Terminal', reward_components['terminal'], episode)

            # Action distribution
            total_actions = sum(self.current_episode_actions.values())
            if total_actions > 0:
                for action, count in self.current_episode_actions.items():
                    self.tb_writer.add_scalar(f'Actions/Action_{action}_Ratio', count / total_actions, episode)

            # Buffer size
            if buffer_size is not None:
                self.tb_writer.add_scalar('Memory/Buffer_Size', buffer_size, episode)

        # Reset episode accumulators
        self._reset_episode_accumulators()

    def _reset_episode_accumulators(self):
        """Reset step-level accumulators for new episode."""
        self.current_episode_losses = []
        self.current_episode_q_expected = []
        self.current_episode_q_targets = []
        self.current_episode_td_errors = []
        self.current_episode_gradient_norms = []
        self.current_episode_param_norms = []
        self.current_episode_actions = defaultdict(int)
        self.episode_start_time = time.time()

    def add_hparam(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters and final metrics to TensorBoard.

        Parameters
        ----------
        hparam_dict : dict
            Hyperparameters (e.g., learning_rate, batch_size, etc.)
        metric_dict : dict
            Final metrics (e.g., final_avg_score, final_success_rate)
        """
        if self.tb_writer:
            self.tb_writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """Close all file handles and writers."""
        if self.enable_csv and self.episode_csv_file:
            self.episode_csv_file.close()
        if self.enable_csv and self.log_step_level and self.step_csv_file:
            self.step_csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()

        print(f"\nLogs saved to: {self.run_dir}")
        if self.enable_csv:
            print(f"  Episode metrics: {self.episode_csv_path}")
            if self.log_step_level:
                print(f"  Step metrics: {self.step_csv_path}")
        if self.enable_tensorboard:
            print(f"  TensorBoard: {self.run_dir / 'tensorboard'}")

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the training run.

        Returns
        -------
        dict
            Summary statistics including average score, success rate, etc.
        """
        return {
            'total_episodes': len(self.episode_scores),
            'total_timesteps': self.global_step,
            'avg_score': np.mean(self.episode_scores),
            'final_avg_score_100': np.mean(self.episode_scores[-100:]) if len(self.episode_scores) >= 100 else np.mean(self.episode_scores),
            'avg_success_rate': np.mean(self.episode_successes),
            'final_success_rate_100': np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes),
            'total_time': time.time() - self.start_time
        }
