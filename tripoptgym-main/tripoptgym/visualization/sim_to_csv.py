"""Simulation to CSV export utilities."""

import gymnasium as gym
import pandas as pd


def create_csv(policy_fn, output_path='simulation.csv', env_id='TripOptWorld-v1',
               route_csv_path=None, start_location=18, end_location=24, verbose=True):
    """Create a CSV of simulation data from an agent navigating the environment.

    Parameters
    ----------
    policy_fn : callable
        Policy function that takes state and returns action
    output_path : str, optional
        Output CSV file path (default: 'simulation.csv')
    env_id : str, optional
        Gymnasium environment ID (default: 'TripOptWorld-v1')
    route_csv_path : str, optional
        Path to route CSV file
    start_location : float, optional
        Start location in miles
    end_location : float, optional
        End location in miles
    verbose : bool, optional
        Print progress (default: True)

    Returns
    -------
    dict
        Statistics: steps, score, etc.

    Examples
    --------
    >>> from tripoptgym.agents.heuristic import heuristic
    >>> stats = create_csv(lambda s: heuristic(None, s), 'demo.csv')

    >>> from tripoptgym.agents.inference import load_trained_agent, trained_agent_policy
    >>> net, dev = load_trained_agent('checkpoint.pth')
    >>> policy = lambda s: trained_agent_policy(net, dev, s)
    >>> stats = create_csv(policy, 'trained.csv')
    """
    env = gym.make(env_id, disable_env_checker=True, render_mode=None,
                   route_csv_path=route_csv_path, start_location=start_location,
                   end_location=end_location)

    sim_data = []
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

        # Extract state information
        train_speed = state[0]
        current_loc = state[2]
        current_speed_limit = state[3]

        # Store simulation data
        sim_data.append({
            'current_loc': current_loc,
            'train_speed': train_speed,
            'Score': env.unwrapped.score
        })

        if verbose:
            # Calculate progress
            distance_covered = current_loc - actual_start
            progress_pct = (distance_covered / total_distance) * 100

            # Print progress (overwrite same line)
            print(f"\rStep {t:3d} | Loc: {current_loc:5.2f}/{actual_end} mi ({progress_pct:5.1f}%) | "
                  f"Speed: {train_speed:4.1f}/{current_speed_limit:4.1f} mph | Score: {env.unwrapped.score:6.1f}",
                  end='', flush=True)

    if verbose:
        print()  # Final newline

    final_score = env.unwrapped.score
    env.close()

    # Save to CSV
    sim_df = pd.DataFrame(sim_data)
    sim_df.to_csv(output_path, index=False)

    stats = {
        'steps': len(sim_data),
        'score': final_score,
        'output_path': output_path
    }

    if verbose:
        print(f"Simulation data saved as '{output_path}' with {len(sim_data)} steps")
        print(f"Final score: {final_score:.1f}")

    return stats
