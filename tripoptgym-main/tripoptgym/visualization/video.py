"""Video generation utilities for creating demonstrations."""

import gymnasium as gym
import imageio
import numpy as np


def create_video(policy_fn, output_path='video.mp4', env_id='TripOptWorld-v1',
                 route_csv_path=None, start_location=18, end_location=24,
                 fps=30, verbose=True):
    """Create a video of an agent navigating the environment.
    
    Parameters
    ----------
    policy_fn : callable
        Policy function that takes state and returns action
    output_path : str, optional
        Output video file path (default: 'video.mp4')
    env_id : str, optional
        Gymnasium environment ID (default: 'TripOptWorld-v1')
    route_csv_path : str, optional
        Path to route CSV file
    start_location : float, optional
        Start location in miles
    end_location : float, optional
        End location in miles
    fps : int, optional
        Frames per second (default: 30)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    dict
        Statistics: frames, score, etc.
        
    Examples
    --------
    >>> from tripoptgym.agents.heuristic import heuristic
    >>> stats = create_video(lambda s: heuristic(None, s), 'demo.mp4')
    
    >>> from tripoptgym.agents.inference import load_trained_agent, trained_agent_policy
    >>> net, dev = load_trained_agent('checkpoint.pth')
    >>> policy = lambda s: trained_agent_policy(net, dev, s)
    >>> stats = create_video(policy, 'trained.mp4')
    """
    env = gym.make(env_id, disable_env_checker=True, render_mode='rgb_array',
                   route_csv_path=route_csv_path, start_location=start_location, 
                   end_location=end_location)
    
    frames = []
    state, _ = env.reset()
    done = False
    t = 0
    
    # Get route info for progress tracking
    actual_start = env.unwrapped.start_location
    actual_end = env.unwrapped.end_location
    total_distance = actual_end - actual_start
    
    while not done:
        action = policy_fn(state)
        t += 1
        state, reward, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        
        if verbose:
            # Extract state information
            train_speed = state[0]
            current_loc = state[2]
            current_speed_limit = state[3]
            
            # Calculate progress
            distance_covered = current_loc - actual_start
            progress_pct = (distance_covered / total_distance) * 100
            
            # Print progress (overwrite same line)
            print(f"\rFrame {t:3d} | Loc: {current_loc:5.2f}/{actual_end} mi ({progress_pct:5.1f}%) | "
                  f"Speed: {train_speed:4.1f}/{current_speed_limit:4.1f} mph | Score: {env.unwrapped.score:6.1f}", 
                  end='', flush=True)
    
    if verbose:
        print()  # Final newline
    
    env.close()
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
    
    stats = {
        'frames': len(frames),
        'score': env.unwrapped.score,
        'output_path': output_path
    }
    
    if verbose:
        print(f"Video saved as '{output_path}' with {len(frames)} frames")
        print(f"Final score: {env.unwrapped.score:.1f}")
    
    return stats
