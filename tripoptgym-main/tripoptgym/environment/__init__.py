"""Environment package for TripOpt Gym."""

from gymnasium.envs.registration import register
from tripoptgym.environment.env import TripOptWorldEnv

# Register the environment with gymnasium
register(
    id="TripOptWorld-v1",
    entry_point="tripoptgym.environment.env:TripOptWorldEnv"
)

__all__ = ["TripOptWorldEnv"]
