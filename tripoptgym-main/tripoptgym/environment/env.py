"""TripOpt Gym environment implementation.

This module contains the main Gymnasium environment for the freight train
control optimization problem.
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd

# Import from within package
from tripoptgym.environment.physics import TrainPhysics, LocomotiveModel
from tripoptgym.visualization.rendering import RollingMap, DataGridView


class TripOptWorldEnv(gym.Env):
    """Freight train optimization environment.
    
    ### Description
    This environment is a freight train optimization problem.

    ### Action Space
    There are 3 discrete actions available:
        0 = do nothing
        1 = notch up
        2 = notch down

    ### Observation Space
    The state is an 6-dimensional vector:
        s[0] is the train speed in mph
        s[1] is the train acceleration in mph/minute
        s[2] is current miles into the route
        s[3] is current speed limit
        s[4] is next speed limit
        s[5] is next speed limit miles into the route

    ### Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased for foward progress along the route, 100 points per mile.
    - is adjusted based on speed limit compliance
    - includes anticipation bonuses for slowing early

    The episode receive an additional reward of -300 for stalling the train (less than 5 mph).
    The episode receive an additional reward of +500 for reaching the end of the route.

    ### Starting State
    The train starts at 7mph in notch 0 (idle).

    ### Episode Termination
    The episode finishes if:
    1) the train stalls (less than 5 mph)
    2) the train reaches the end of the route

    ### Version History
    - v1: Updated reward structure

    ### Credits
    Created by Joe Wakeman, 2024
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, route_csv_path, render_mode=None, start_location=18, end_location=24):
        """Initialize the TripOpt World environment.
        
        Parameters
        ----------
        route_csv_path : str
            Path to route CSV file (required)
        render_mode : str, optional
            Rendering mode ('human' or 'rgb_array')
        start_location : float, optional
            Starting location in miles (default: 18)
        end_location : float, optional
            Ending location in miles (default: 24)
        """
        pygame.init()
        
        # Load route data from local file
        self.route_data = pd.read_csv(route_csv_path)
        
        self.rollingmap = RollingMap(self.route_data)
        self.custom_data_monitor = DataGridView(pygame)
        self.physics = TrainPhysics()
        self.loco_model = LocomotiveModel()
        self.start_location = start_location
        self.end_location = end_location
        self.locoLocation = self.start_location
        self.locoSpeed = 7
        self.locoNotch = 0
        self.time = 0
        self.last_notch_time = 0
        self.spd_limit = 0
        self.nxt_spd_limit = 0
        self.nxt_spd_limit_location = 0

        self.train_mass = 7727
        self.davis_a = 1.552524209022522
        self.davis_b = 0.01162271574139595
        self.davis_c = 0.00077766168396919966

        self.score = 0

        # Performance optimization: track last indices for route data lookups
        self.last_grade_index = 0
        self.last_speed_limit_index = 0
        self.last_next_speed_limit_index = 0

        # We have 3 actions, corresponding to "hold notch", "notch up", "notch down"
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 6-dimensional continuous state
        # [speed, acceleration, location, current_speed_limit, next_speed_limit, next_speed_limit_location]
        self.observation_space = spaces.Box(
            low=np.array([0, -100, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 100, 100, 100, 100], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional reset options
            
        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional information
        """
        self.rollingmap.reset()
        self.score = 0
        self.locoLocation = self.start_location
        self.locoSpeed = 7
        self.locoNotch = 0
        self.time = 0
        self.last_notch_time = 0
        self.spd_limit = 0
        self.nxt_spd_limit = 0
        self.nxt_spd_limit_location = 0

        # Reset performance optimization indices
        self.last_grade_index = 0
        self.last_speed_limit_index = 0
        self.last_next_speed_limit_index = 0

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.spd_limit = self.speedLimitAtDir(self.locoLocation)
        self.nxt_spd_limit_location, self.nxt_spd_limit = self.nextSpeedLimitChange(self.locoLocation)

        state = [
            self.locoSpeed,
            0,
            self.locoLocation,
            self.spd_limit,
            self.nxt_spd_limit,
            self.nxt_spd_limit_location,
        ]
        assert len(state) == 6

        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        """Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Action to take (0=hold, 1=notch up, 2=notch down)
            
        Returns
        -------
        observation : np.ndarray
            Next observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode has terminated
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information including reward components
        """
        # enforce 3 seconds between notch changes
        if self.time - self.last_notch_time > 2.9:
            if action == 0:
                pass
            elif action == 1:
                self.locoNotch = min(8, self.locoNotch + 1)
                self.last_notch_time = self.time
            elif action == 2:
                self.locoNotch = max(-8, self.locoNotch - 1)
                self.last_notch_time = self.time

        h = self.locoSpeed / 3600
        g = self.gradeAtDir(self.locoLocation)
        gplus = self.gradeAtDir(self.locoLocation + h)
        p = self.loco_model.THPForNotch(self.locoNotch)
        pplus = self.loco_model.THPForNotch(self.locoNotch)
        Fabe = 0
        FabePlus = 0
        Fsat = self.loco_model.MaxTEForNotch(self.locoNotch)
        FsatPlus = self.loco_model.MaxTEForNotch(self.locoNotch)
        cbs = self.loco_model.CommutationBreakpointSpeed()
        hpmaxspd = self.loco_model.MaxCommutationBrakingHP()
        maxbhp = self.loco_model.MaxDynamicBrakingHP()

        vplus, pplus, fail_code = self.physics.trapz_integrate_train_one_step(
                                                    h, self.locoSpeed, g, gplus,
                                                    self.train_mass, self.davis_a, self.davis_b, self.davis_c,
                                                    p, pplus, Fabe, FabePlus,
                                                    Fsat, FsatPlus, 0, cbs, hpmaxspd, maxbhp)

        dv = vplus - self.locoSpeed
        dt = (1/self.locoSpeed)*h*60 # minutes

        self.time = self.time + (dt * 60) # seconds

        self.locoLocation = self.locoLocation + h
        self.locoSpeed = vplus

        self.spd_limit = self.speedLimitAtDir(self.locoLocation)
        self.nxt_spd_limit_location, self.nxt_spd_limit = self.nextSpeedLimitChange(self.locoLocation)

        state = [
            self.locoSpeed,
            dv/dt,
            self.locoLocation,
            self.spd_limit,
            self.nxt_spd_limit,
            self.nxt_spd_limit_location,
        ]
        assert len(state) == 6

        # Initialize reward components for tracking
        reward_progress = 0
        reward_speed_compliance = 0
        reward_anticipation = 0
        reward_terminal = 0
        
        # 1. BASE REWARD: Scale progress reward by distance
        # h is in miles traveled this step
        reward_progress = 100 * h  # Increased from 10 to 100 for stronger signal
        
        # 2. SPEED LIMIT COMPLIANCE: Continuous, symmetric penalty with acceleration awareness
        speed_error = self.locoSpeed - self.spd_limit
        acceleration = dv/dt  # mph per minute
        
        if speed_error > 0:  # Over speed limit
            # Quadratic penalty for overspeed (gets expensive fast)
            if speed_error <= 3:
                reward_speed_compliance = -5 * (speed_error ** 1.5)
            else:
                reward_speed_compliance = -50 * speed_error  # Severe but not overwhelming
        else:  # Under speed limit
            # Encourage speed optimization but don't heavily penalize safe/conservative driving
            speed_ratio = self.locoSpeed / max(self.spd_limit, 1.0)
            
            # If accelerating well while below speed limit, reward the correct behavior
            if speed_ratio < 0.85 and acceleration > 5.0:  # Accelerating at good rate
                reward_speed_compliance = 0.5 * acceleration  # Modest reward for acceleration effort
            elif speed_ratio < 0.4:  # Very slow and not accelerating much
                if acceleration > 2.0:  # But still trying to accelerate
                    reward_speed_compliance = 0  # Neutral - give it time to speed up
                else:
                    reward_speed_compliance = -5 * (0.4 - speed_ratio) * self.spd_limit
            elif speed_ratio >= 0.85:  # Reward getting close to speed limit
                reward_speed_compliance = 10 * (speed_ratio - 0.85) * 10  # Bonus for efficient speed
            else:  # Acceptable speed range (40-85% of limit)
                reward_speed_compliance = 0  # Neutral, no penalty
        
        # 3. ANTICIPATION BONUS: Reward slowing down early for upcoming limits
        distance_to_next_limit = self.nxt_spd_limit_location - self.locoLocation
        if self.nxt_spd_limit < self.spd_limit and distance_to_next_limit < 2.0:
            # Approaching a lower speed limit
            target_speed = self.nxt_spd_limit + 5  # Allow small buffer
            if self.locoSpeed <= target_speed:
                reward_anticipation = 10 * (1 - distance_to_next_limit / 2.0)
        
        # 4. TERMINAL REWARDS: Reduced magnitude for better balance
        termination_reason = 0  # 0=not terminated, 1=destination reached, 2=stalled, 3=overspeed
        
        if self.locoLocation > self.end_location:
            termination_reason = 1
            reward_terminal = 500  # Reduced from 1000
        
        # Train stalled
        if termination_reason == 0 and self.locoSpeed < 5:
            termination_reason = 2
            reward_terminal = -300  # Reduced from 1000
        
        # Total reward
        reward = reward_progress + reward_speed_compliance + reward_anticipation + reward_terminal
        self.score = self.score + reward
        
        terminated = (termination_reason > 0)

        return np.array(state, dtype=np.float32), reward, terminated, False, {
            'termination_reason': termination_reason,
            'reward_progress': reward_progress,
            'reward_speed_compliance': reward_speed_compliance,
            'reward_anticipation': reward_anticipation,
            'reward_terminal': reward_terminal
        }

    def gradeAtDir(self, value):
        """Get grade at a specific location via interpolation.

        Parameters
        ----------
        value : float
            Location in miles

        Returns
        -------
        float
            Grade percent at location
        """
        # Start from last known index for performance
        for i in range(self.last_grade_index, len(self.route_data)):
            row = self.route_data.iloc[i]
            if row['Distance In Route'] > value:
                if i > 0:
                    self.last_grade_index = i - 1  # Update for next call
                    x = [self.route_data.iloc[i - 1]['Distance In Route'], self.route_data.iloc[i]['Distance In Route']]
                    y = [self.route_data.iloc[i - 1]['Effective Grade Percent'], self.route_data.iloc[i]['Effective Grade Percent']]
                    return np.interp(value, x, y)
                break
        return 0

    def speedLimitAtDir(self, value):
        """Get speed limit at a specific location.

        Parameters
        ----------
        value : float
            Location in miles

        Returns
        -------
        float
            Speed limit in mph at location
        """
        # Start from last known index for performance
        for i in range(self.last_speed_limit_index, len(self.route_data)):
            row = self.route_data.iloc[i]
            if row['Distance In Route'] > value:
                if i > 0:
                    self.last_speed_limit_index = i - 1  # Update for next call
                    return self.route_data.iloc[i - 1]['Effective Speed Limit']
                break
        return 0

    def nextSpeedLimitChange(self, current_location):
        """Find the next speed limit change location.

        Parameters
        ----------
        current_location : float
            Current location in miles

        Returns
        -------
        next_location : float
            Location of next speed limit change
        next_speed_limit : float
            Value of next speed limit
        """
        current_speed_limit = self.speedLimitAtDir(current_location)
        next_location = current_location
        next_speed_limit = current_speed_limit
        # Start from last known index for performance
        for i in range(self.last_next_speed_limit_index, len(self.route_data)):
            row = self.route_data.iloc[i]
            if row['Distance In Route'] > current_location:
                if row['Effective Speed Limit'] != current_speed_limit:
                    self.last_next_speed_limit_index = i  # Update for next call
                    next_location = row['Distance In Route']
                    next_speed_limit = row['Effective Speed Limit']
                    break

        return next_location, next_speed_limit

    def render(self):
        """Render the environment.
        
        Returns
        -------
        np.ndarray
            RGB array of rendered frame
        """
        return self._render_frame()

    def _render_frame(self):
        """Generate a single frame for rendering.
        
        Returns
        -------
        np.ndarray
            RGB array of rendered frame
        """
        if not pygame.get_init() == True:
            pygame.init()

        canvas = self.rollingmap.update(pygame, self.locoLocation, self.locoSpeed)
        self.custom_data_monitor.DrawGrid(canvas)
        self.custom_data_monitor.DrawValue("Speed", 1, 1, self.locoSpeed, canvas)
        self.custom_data_monitor.DrawValue("Notch", 1, 2, self.locoNotch, canvas)
        self.custom_data_monitor.DrawValue("Location", 1, 3, self.locoLocation, canvas)
        self.custom_data_monitor.DrawValue("Score", 1, 4, self.score, canvas)
        self.custom_data_monitor.DrawValue("Speed Limit", 2, 1, self.spd_limit, canvas)
        self.custom_data_monitor.DrawValue("NxtSpdLmt", 2, 2, self.nxt_spd_limit, canvas)
        self.custom_data_monitor.DrawValue("NextSpdLmtLoc", 2, 3, self.nxt_spd_limit_location, canvas)

        title = "TripOpt Gym -- Created by Joe Wakeman, 2024"
        title_font = pygame.font.SysFont('calibri', 14)
        title_width, title_height = title_font.size(title)
        title_color = (210, 130, 0)
        title_surface = title_font.render(title, False, title_color)
        title_rect = title_surface.get_rect()
        title_rect.topleft = (720 - title_width - 10, 250)
        canvas.blit(title_surface, title_rect)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        """Clean up resources."""
        pygame.quit()
