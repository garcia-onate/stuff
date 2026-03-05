#!/usr/bin/env python3
"""
TripOptGym CLI Driver

Unified command-line interface for training, demonstration, and route conversion.
"""

import argparse
import sys
import os
from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np

import tripoptgym.environment  # Register the gym environment
from tripoptgym.agents.dqn import Agent
from tripoptgym.agents.heuristic import heuristic
from tripoptgym.agents.inference import load_trained_agent, trained_agent_policy
from tripoptgym.visualization.video import create_video
from tripoptgym.visualization.sim_to_csv import create_csv
from tripoptgym.utils.config import load_config
from tripoptgym.utils.device import get_device, get_device_name
from tripoptgym.utils.route_converter import convert_route_data
from tripoptgym.utils.process_dr_data import process_dr_data
from tripoptgym.utils.training_logger import TrainingLogger


def train_command(args):
    """Execute training workflow."""
    print("=" * 60)
    print("TripOptGym Training")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from: {args.config}")
    
    # Use config values if command-line args not specified
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    start_location = args.start if args.start is not None else env_config.get('start_location', 0.0)
    end_location = args.end if args.end is not None else env_config.get('end_location', None)
    num_episodes = args.episodes if args.episodes is not None else training_config.get('number_episodes', 1000)
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else training_config.get('checkpoint_interval', 100)
    
    # Override device if specified
    if args.device:
        device = get_device(args.device)
        print(f"Device: {get_device_name(device)} (user specified: {args.device})")
    else:
        device = get_device(config.get('device', {}).get('preference', 'auto'))
        print(f"Device: {get_device_name(device)}")
    
    # Create environment(s)
    num_parallel_envs = config.get('environment', {}).get('num_parallel_envs', 1)
    print(f"\nCreating {num_parallel_envs} parallel environment(s) with route: {args.csv}")
    
    if num_parallel_envs > 1:
        # Create vectorized environment for parallel execution
        env = AsyncVectorEnv([
            lambda: gym.make(
                'TripOptWorld-v1',
                route_csv_path=args.csv,
                start_location=start_location,
                end_location=end_location
            )
            for _ in range(num_parallel_envs)
        ])
        # Get a single environment for metadata
        single_env = gym.make(
            'TripOptWorld-v1',
            route_csv_path=args.csv,
            start_location=start_location,
            end_location=end_location
        )
        state_size = single_env.observation_space.shape[0]
        action_size = single_env.action_space.n
        start_location = single_env.unwrapped.start_location
        end_location = single_env.unwrapped.end_location
        single_env.close()
    else:
        # Single environment
        env = gym.make(
            'TripOptWorld-v1',
            route_csv_path=args.csv,
            start_location=start_location,
            end_location=end_location
        )
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        start_location = env.unwrapped.start_location
        end_location = env.unwrapped.end_location
    agent = Agent(
        state_size, 
        action_size,
        learning_rate=config['training']['learning_rate'],
        replay_buffer_size=config['training']['replay_buffer_size'],
        device=device,
        hidden_layers=config['network']['hidden_layers'],
        update_frequency=config['training'].get('update_frequency', 4)
    )
    
    print(f"\nAgent configuration:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Minibatch size: {config['training']['minibatch_size']}")
    print(f"  Epsilon: {config['training']['epsilon_start']} -> {config['training']['epsilon_end']}")
    print(f"  Decay: {config['training']['epsilon_decay']}")
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)

    # Initialize logger
    logging_config = config.get('logging', {})
    logger = None
    if logging_config.get('enable_logging', True):
        logger = TrainingLogger(
            log_dir=logging_config.get('log_dir', 'logs'),
            run_name=logging_config.get('run_name'),
            enable_csv=logging_config.get('enable_csv', True),
            enable_tensorboard=logging_config.get('enable_tensorboard', True),
            log_step_level=logging_config.get('log_step_level', False)
        )
        print(f"\nLogging enabled: {logger.run_dir}")

        # Log hyperparameters
        hparams = {
            'learning_rate': config['training']['learning_rate'],
            'minibatch_size': config['training']['minibatch_size'],
            'discount_factor': config['training']['discount_factor'],
            'epsilon_start': config['training']['epsilon_start'],
            'epsilon_end': config['training']['epsilon_end'],
            'epsilon_decay': config['training']['epsilon_decay'],
            'replay_buffer_size': config['training']['replay_buffer_size'],
            'num_parallel_envs': num_parallel_envs,
            'hidden_layers': str(config['network']['hidden_layers'])
        }

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    all_episode_scores = []
    epsilon_log = []
    # Initialize epsilon from config
    epsilon = config['training']['epsilon_start']
    epsilon_decay = config['training']['epsilon_decay']
    epsilon_min = config['training']['epsilon_end']
    
    # Get route info for display
    total_distance = end_location - start_location
    
    for episode in range(1, num_episodes + 1):
        if num_parallel_envs > 1:
            # Parallel environment training
            states, _ = env.reset()
            episode_scores = np.zeros(num_parallel_envs)
            dones = np.zeros(num_parallel_envs, dtype=bool)
            termination_codes = np.zeros(num_parallel_envs, dtype=int)
            step_counts = np.zeros(num_parallel_envs, dtype=int)
            
            # Track reward components for each env
            reward_components = {
                'progress': np.zeros(num_parallel_envs),
                'speed_compliance': np.zeros(num_parallel_envs),
                'anticipation': np.zeros(num_parallel_envs),
                'terminal': np.zeros(num_parallel_envs)
            }
            
            # Store final state for each env when it terminates
            final_states = [None] * num_parallel_envs
            final_scores = [None] * num_parallel_envs
            final_frames = [None] * num_parallel_envs
            episode_finished = False
            
            max_steps = config['training'].get('max_timesteps', 25000)
            for t in range(max_steps):
                # Select actions for all environments
                actions = np.array([agent.act(states[i], epsilon) if not dones[i] else 0
                                   for i in range(num_parallel_envs)])
                
                next_states, rewards, new_dones, _, infos = env.step(actions)
                
                # Store experiences and update for each environment
                for i in range(num_parallel_envs):
                    if not dones[i]:
                        # Capture learning metrics
                        metrics = agent.step(states[i], actions[i], rewards[i], next_states[i], new_dones[i],
                                 minibatch_size=config['training']['minibatch_size'],
                                 discount_factor=config['training']['discount_factor'],
                                 interpolation_parameter=config['training']['interpolation_parameter'])

                        # Log step-level metrics if learning occurred
                        if logger and metrics is not None:
                            logger.log_step(
                                episode=episode,
                                step=step_counts[i],
                                action=actions[i],
                                loss=metrics['loss'],
                                q_expected=metrics['q_expected_mean'],
                                q_target=metrics['q_target_mean'],
                                td_error=metrics['td_error'],
                                gradient_norm=metrics['gradient_norm'],
                                param_norm=metrics['param_norm'],
                                buffer_size=len(agent.memory.memory)
                            )

                        episode_scores[i] += rewards[i]
                        step_counts[i] += 1
                        
                        # Track reward components
                        if 'reward_progress' in infos:
                            reward_components['progress'][i] += infos['reward_progress'][i]
                            reward_components['speed_compliance'][i] += infos['reward_speed_compliance'][i]
                            reward_components['anticipation'][i] += infos['reward_anticipation'][i]
                            reward_components['terminal'][i] += infos['reward_terminal'][i]
                        
                        # Save final state when environment finishes
                        if new_dones[i]:
                            if 'termination_reason' in infos:
                                termination_codes[i] = infos['termination_reason'][i]
                            final_states[i] = next_states[i].copy()
                            final_scores[i] = episode_scores[i]
                            final_frames[i] = step_counts[i]
                
                states = next_states
                dones = np.logical_or(dones, new_dones)
                
                # Display progress for all environments (update lines in place)
                # Move cursor up (num_parallel_envs lines if not first iteration)
                if t > 0:
                    print(f'\033[{num_parallel_envs}A', end='')  # Move cursor up num_parallel_envs lines
                
                termination_labels = {0: "    ", 1: "DEST", 2: "STAL", 3: "OVER"}
                for i in range(num_parallel_envs):
                    # Use final state if environment has terminated, otherwise use current state
                    if dones[i] and final_states[i] is not None:
                        display_state = final_states[i]
                        display_score = final_scores[i]
                        display_frame = final_frames[i]
                    else:
                        display_state = next_states[i]
                        display_score = episode_scores[i]
                        display_frame = step_counts[i]
                    
                    train_speed = display_state[0]
                    current_loc = display_state[2]
                    current_speed_limit = display_state[3]
                    distance_covered = current_loc - start_location
                    progress_pct = (distance_covered / total_distance) * 100
                    status = termination_labels[termination_codes[i]] if dones[i] else "    "
                    
                    print(f"\rEnv {i+1} | Ep {episode:4d} | Step {display_frame:3d} | Loc: {current_loc:5.2f}/{end_location:.0f} ({progress_pct:5.1f}%) | "
                          f"Speed: {train_speed:4.1f}/{current_speed_limit:4.1f} | Score: {display_score:7.1f} {status}\033[K")
                
                print(f"Average Score: {np.mean(episode_scores):7.1f}", end='', flush=True)
                
                # Check if all environments are done
                if np.all(dones):
                    # Print termination reasons
                    termination_reasons = {0: "Unknown", 1: "Destination", 2: "Stalled", 3: "Overspeed"}
                    reasons = [termination_reasons[code] for code in termination_codes]
                    reason_counts = {r: reasons.count(r) for r in set(reasons) if r != "Unknown"}
                    print(f" - Terminations: {reason_counts}")
                    episode_finished = True
                    break
            
            # Print newlines after episode completion
            if not episode_finished:
                print("\n" * 1)
            else:
                print()
            
            # Episode summary for parallel
            mean_score = np.mean(episode_scores)
            termination_reasons = {0: "Unknown", 1: "Destination", 2: "Stalled", 3: "Overspeed"}
            reasons = [termination_reasons[code] for code in termination_codes]
            reason_counts = {r: reasons.count(r) for r in set(reasons)}
            print(f"Ep {episode:4d} Complete | Avg Score: {mean_score:7.1f} | Terminations: {reason_counts}")
            
            all_episode_scores.append(mean_score)

            # Log episode-level metrics (aggregate across parallel envs)
            if logger:
                # Determine primary termination reason (most common)
                primary_termination = int(np.bincount(termination_codes.astype(int)).argmax())
                avg_reward_components = {
                    'progress': float(np.mean(reward_components['progress'])),
                    'speed_compliance': float(np.mean(reward_components['speed_compliance'])),
                    'anticipation': float(np.mean(reward_components['anticipation'])),
                    'terminal': float(np.mean(reward_components['terminal']))
                }
                logger.log_episode(
                    episode=episode,
                    score=mean_score,
                    steps=int(np.mean(step_counts)),
                    epsilon=epsilon,
                    reward_components=avg_reward_components,
                    termination_reason=primary_termination,
                    buffer_size=len(agent.memory.memory)
                )

            # Print reward breakdown
            print(f"  Rewards: Prog={np.mean(reward_components['progress']):6.1f}, "
                  f"Speed={np.mean(reward_components['speed_compliance']):6.1f}, "
                  f"Antic={np.mean(reward_components['anticipation']):5.1f}, "
                  f"Term={np.mean(reward_components['terminal']):6.1f}")
            
        else:
            # Single environment training
            state, _ = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            # Track reward components
            reward_components_single = {
                'progress': 0.0,
                'speed_compliance': 0.0,
                'anticipation': 0.0,
                'terminal': 0.0
            }
            termination_reason = 0
            
            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Capture learning metrics
                metrics = agent.step(state, action, reward, next_state, done,
                         minibatch_size=config['training']['minibatch_size'],
                         discount_factor=config['training']['discount_factor'],
                         interpolation_parameter=config['training']['interpolation_parameter'])

                # Log step-level metrics if learning occurred
                if logger and metrics is not None:
                    logger.log_step(
                        episode=episode,
                        step=step_count,
                        action=action,
                        loss=metrics['loss'],
                        q_expected=metrics['q_expected_mean'],
                        q_target=metrics['q_target_mean'],
                        td_error=metrics['td_error'],
                        gradient_norm=metrics['gradient_norm'],
                        param_norm=metrics['param_norm'],
                        buffer_size=len(agent.memory.memory)
                    )

                # Track reward components
                if 'reward_progress' in info:
                    reward_components_single['progress'] += info['reward_progress']
                    reward_components_single['speed_compliance'] += info['reward_speed_compliance']
                    reward_components_single['anticipation'] += info['reward_anticipation']
                    reward_components_single['terminal'] += info['reward_terminal']

                if 'termination_reason' in info:
                    termination_reason = info['termination_reason']

                # Real-time progress display
                train_speed = next_state[0]
                current_loc = next_state[2]
                current_speed_limit = next_state[3]
                distance_covered = current_loc - start_location
                progress_pct = (distance_covered / total_distance) * 100

                print(f"\rEp {episode:4d} | Step {step_count:3d} | "
                      f"Loc: {current_loc:5.2f}/{end_location:.0f} mi ({progress_pct:5.1f}%) | "
                      f"Speed: {train_speed:4.1f}/{current_speed_limit:4.1f} mph | "
                      f"Score: {total_reward:7.1f}", end='', flush=True)

                state = next_state
                total_reward += reward
                step_count += 1
            
            # Clear the real-time line and print final episode info
            termination_labels = {0: "Unknown", 1: "Destination", 2: "Stalled", 3: "Overspeed"}
            print(f"\r{'':100}", end='')  # Clear line
            print(f"\rEp {episode:4d} | Score: {total_reward:7.1f} | Steps: {step_count:3d} | "
                  f"Term: {termination_labels[termination_reason]}")
            
            all_episode_scores.append(total_reward)

            # Log episode-level metrics
            if logger:
                logger.log_episode(
                    episode=episode,
                    score=total_reward,
                    steps=step_count,
                    epsilon=epsilon,
                    reward_components=reward_components_single,
                    termination_reason=termination_reason,
                    buffer_size=len(agent.memory.memory)
                )

            # Print reward breakdown
            print(f"  Rewards: Prog={reward_components_single['progress']:6.1f}, "
                  f"Speed={reward_components_single['speed_compliance']:6.1f}, "
                  f"Antic={reward_components_single['anticipation']:5.1f}, "
                  f"Term={reward_components_single['terminal']:6.1f}")
        
        epsilon_log.append(epsilon)
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        # Print rolling average and epsilon (common for both single and parallel)
        avg_score = np.mean(all_episode_scores[-100:]) if len(all_episode_scores) >= 100 else np.mean(all_episode_scores)
        print(f"  Avg(100): {avg_score:7.2f} | Epsilon: {epsilon:.4f}")
        
        # Save checkpoint
        if episode % checkpoint_interval == 0:
            print(f"  {'='*60}")
            print(f"  Checkpoint {episode} | Avg Score: {avg_score:.2f}")
            print(f"  {'='*60}")
            checkpoint_path = f"checkpoint_ep{episode}.pth"
            agent.save_checkpoint(checkpoint_path, episode, epsilon, all_episode_scores)
            print(f"  --> Saved: {checkpoint_path}\n")
        
        # Check for early termination if target score is set
        if args.target_score is not None and len(all_episode_scores) >= 100:
            avg_score = np.mean(all_episode_scores[-100:])
            if avg_score >= args.target_score:
                print(f"\n{'='*60}")
                print(f"Environment solved at episode {episode}!")
                print(f"Average Score: {avg_score:.2f} (target: {args.target_score:.2f})")
                print(f"{'='*60}")
                final_path = "train_checkpoint.pth"
                agent.save_checkpoint(final_path, episode, epsilon, all_episode_scores)
                print(f"Final model saved to: {final_path}")

                # Finalize logger
                if logger:
                    summary_stats = logger.get_summary_stats()
                    if logger.enable_tensorboard:
                        # Log final hyperparameters and metrics
                        final_metrics = {
                            'final_avg_score_100': summary_stats['final_avg_score_100'],
                            'final_success_rate_100': summary_stats['final_success_rate_100']
                        }
                        logger.add_hparam(hparams, final_metrics)
                    logger.close()

                env.close()
                return

    # Save final model
    final_path = "train_checkpoint.pth"
    agent.save_checkpoint(final_path, num_episodes, epsilon, all_episode_scores)
    print(f"\n{'='*60}")
    print(f"Training complete! Final model saved to: {final_path}")
    print(f"{'='*60}")

    # Finalize logger and print summary
    if logger:
        summary_stats = logger.get_summary_stats()
        print(f"\nTraining Summary:")
        print(f"  Total episodes: {summary_stats['total_episodes']}")
        print(f"  Total timesteps: {summary_stats['total_timesteps']}")
        print(f"  Avg score (all): {summary_stats['avg_score']:.2f}")
        print(f"  Avg score (last 100): {summary_stats['final_avg_score_100']:.2f}")
        print(f"  Success rate: {summary_stats['avg_success_rate']:.2%}")
        print(f"  Total time: {summary_stats['total_time']:.1f}s")

        if logger.enable_tensorboard:
            # Log final hyperparameters and metrics
            final_metrics = {
                'final_avg_score_100': summary_stats['final_avg_score_100'],
                'final_success_rate_100': summary_stats['final_success_rate_100']
            }
            logger.add_hparam(hparams, final_metrics)
        logger.close()

    env.close()


def demo_command(args):
    """Execute demonstration workflow."""
    print("=" * 60)
    print("TripOptGym Demonstration")
    print("=" * 60)
    
    # Load config if available for default values
    config_path = Path('configs/default_config.yaml')
    if config_path.exists():
        config = load_config(str(config_path))
        env_config = config.get('environment', {})
        start_location = args.start if args.start is not None else env_config.get('start_location', 0.0)
        end_location = args.end if args.end is not None else env_config.get('end_location', None)
    else:
        start_location = args.start if args.start is not None else 0.0
        end_location = args.end
    
    # Override device if specified
    if args.device:
        device = get_device(args.device)
        print(f"\nDevice: {get_device_name(device)} (user specified: {args.device})")
    else:
        device = get_device()
        print(f"\nDevice: {get_device_name(device)}")
    
    # Select agent policy
    if args.agent == 'heuristic':
        print("Agent: Heuristic policy")
        agent_policy = lambda state: heuristic(None, state)
        model_path = None
    elif args.agent == 'trained':
        if not args.model or not os.path.exists(args.model):
            print(f"Error: Trained model not found at: {args.model}")
            sys.exit(1)
        print(f"Agent: Trained DQN from {args.model}")
        model_path = args.model
        
        # Load the agent
        env_temp = gym.make('TripOptWorld-v1', route_csv_path=args.csv, start_location=start_location, end_location=end_location)
        state_size = env_temp.observation_space.shape[0]
        action_size = env_temp.action_space.n
        env_temp.close()
        
        agent_model, model_device = load_trained_agent(model_path, state_size, action_size, device)
        agent_policy = lambda state: trained_agent_policy(agent_model, model_device, state)
    else:
        print(f"Error: Unknown agent type: {args.agent}")
        sys.exit(1)
    
    # Create and run demo
    print(f"\nRoute: {args.csv}")
    print(f"Output: {args.output}")

    # Determine output type
    if args.output_type:
        output_type = args.output_type
    else:
        # Auto-detect from file extension
        output_type = 'csv' if args.output.endswith('.csv') else 'video'

    if output_type == 'csv':
        print("Creating CSV simulation data...")
        stats = create_csv(
            policy_fn=agent_policy,
            output_path=args.output,
            route_csv_path=args.csv,
            start_location=start_location,
            end_location=end_location
        )

        print(f"\n{'='*60}")
        print(f"Demo complete! CSV saved to: {args.output}")
        print(f"Timesteps: {stats['steps']}")
        print(f"Final score: {stats['score']:.1f}")
        print(f"{'='*60}")
    else:
        print("Creating demonstration video...")
        stats = create_video(
            policy_fn=agent_policy,
            output_path=args.output,
            route_csv_path=args.csv,
            start_location=start_location,
            end_location=end_location,
            fps=30
        )

        print(f"\n{'='*60}")
        print(f"Demo complete! Video saved to: {args.output}")
        print(f"Timesteps: {stats['frames']}")
        print(f"Final score: {stats['score']:.1f}")
        print(f"{'='*60}")


def convert_command(args):
    """Execute route data conversion."""
    print("=" * 60)
    if args.type == 'rtc':
        print("RTC Route Text File Conversion")
    else:
        print("Data Recorder File Conversion")
    print("=" * 60)
    
    print(f"\nInput:  {args.input_file}")
    print(f"Output: {args.output_file}\n")
    
    try:
        if args.type == 'rtc':
            convert_route_data(args.input_file, args.output_file)
        else:  # data recorder
            process_dr_data(args.input_file, args.output_file)
        print(f"\n{'='*60}")
        print("Conversion complete!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TripOptGym - Train Trip Optimization Reinforcement Learning Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent
  tripopt train --csv route_data.csv --episodes 1000
  
  # Resume training from checkpoint
  tripopt train --csv route_data.csv --resume checkpoint_ep500.pth --episodes 1000
  
  # Demo with heuristic agent
  tripopt demo --csv route_data.csv --agent heuristic --output demo_heuristic.mp4
  
  # Demo with trained agent
  tripopt demo --csv route_data.csv --agent trained --model train_checkpoint.pth --output demo_trained.mp4
  
  # Convert data recorder file (default)
  tripopt convert dr_input.csv dr_output.csv
  
  # Convert RTC route text file
  tripopt convert --type rtc route_input_parsed.txt route_data_generated.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a DQN agent')
    train_parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                              help='Path to configuration YAML file (default: configs/default_config.yaml)')
    train_parser.add_argument('--csv', type=str, required=True,
                              help='Path to route CSV file')
    train_parser.add_argument('--start', type=float, default=None,
                              help='Starting distance (miles, default: from config or 0.0)')
    train_parser.add_argument('--end', type=float, default=None,
                              help='Ending distance (miles, default: None = end of route)')
    train_parser.add_argument('--episodes', type=int, default=None,
                              help='Number of training episodes (default: from config or 1000)')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint file to resume from')
    train_parser.add_argument('--checkpoint-interval', type=int, default=None,
                              help='Save checkpoint every N episodes (default: from config or 100)')
    train_parser.add_argument('--target-score', type=float, default=None,
                              help='Target 100-episode average score for early termination (default: None = no early termination)')
    train_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default=None,
                              help='Device preference (overrides config file)')
    train_parser.set_defaults(func=train_command)
    
    # Demo subcommand
    demo_parser = subparsers.add_parser('demo', help='Run demonstration and create video')
    demo_parser.add_argument('--csv', type=str, required=True,
                             help='Path to route CSV file')
    demo_parser.add_argument('--start', type=float, default=None,
                             help='Starting distance (miles, default: from config or 0.0)')
    demo_parser.add_argument('--end', type=float, default=None,
                             help='Ending distance (miles, default: None = end of route)')
    demo_parser.add_argument('--agent', type=str, choices=['heuristic', 'trained'], required=True,
                             help='Agent type: heuristic or trained')
    demo_parser.add_argument('--model', type=str, default=None,
                             help='Path to trained model checkpoint (required for --agent trained)')
    demo_parser.add_argument('--output', type=str, default='demo.mp4',
                             help='Output file path (default: demo.mp4). Use .csv extension for CSV output, .mp4 for video.')
    demo_parser.add_argument('--output-type', type=str, choices=['video', 'csv'], default=None,
                             help='Output type: video or csv. If not specified, auto-detects from output file extension.')
    demo_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default=None,
                             help='Device preference')
    demo_parser.set_defaults(func=demo_command)
    
    # Convert subcommand
    convert_parser = subparsers.add_parser('convert', help='Convert route data to CSV')
    convert_parser.add_argument('--type', type=str, choices=['rtc', 'dr'], default='dr',
                                help='Conversion type: rtc (parsed route text file) or dr (data recorder CSV). Default: dr')
    convert_parser.add_argument('input_file', type=str,
                                help='Input file (parsed route text or data recorder CSV)')
    convert_parser.add_argument('output_file', type=str,
                                help='Output CSV file')
    convert_parser.set_defaults(func=convert_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
