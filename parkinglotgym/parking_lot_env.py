import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from parkinglot.lot import Lot


class ParkingLotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, layout_str: str):
        super(ParkingLotEnv, self).__init__()

        # Initialize the parking lot
        self.lot = Lot(layout_str)
        self.initial_layout = layout_str

        # Define action space: (vehicle_id, move)
        # vehicle_id is an integer index into the list of vehicles
        self.vehicle_ids = list(self.lot.query_vehicles().keys())

        # Action space: (vehicle_index, move)
        # vehicle_index: which vehicle to move (discrete)
        # move: how many steps to move (discrete)
        self.width, self.height = self.lot.dimensions()
        self.action_space = spaces.MultiDiscrete([len(self.vehicle_ids), max(self.width, self.height)])

        # Observation space: grid state
        # Each cell can be: empty (0), wall (1), or vehicle (2+)
        self.observation_space = spaces.Box(
            low=0,
            high=len(self.vehicle_ids) + 1,  # +1 for walls
            shape=(self.height, self.width),
            dtype=np.int32
        )

        self.reset()

    def _get_available_moves(self) -> Dict[str, Tuple[int, int]]:
        """Get available moves for each vehicle."""
        return self.lot.query_legal_moves()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.lot = Lot(self.initial_layout)
        observation = self._get_observation()
        info = {
            'available_moves': self._get_available_moves(),
        }
        return observation, info

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        vehicle_id_num, move = action
        if vehicle_id_num < 2 or vehicle_id_num >= len(self.vehicle_ids) + 2:
            raise ValueError(
                f"Invalid vehicle ID: {vehicle_id_num}. Vehicles start at index 2 and go up to {len(self.vehicle_ids) + 1}.")
        vehicle_id = self.vehicle_ids[vehicle_id_num - 2]

        try:
            # Attempt to make the move
            self.lot.move(vehicle_id, move)

            # Check if the game is solved
            done = self.lot.is_solved()
            reward = 0 if done else -1  # Small negative reward for each move

        except ValueError as _:
            # Invalid move
            done = False
            reward = -2

        # Get the new observation
        observation = self._get_observation()

        # Additional info
        info = {
            'available_moves': self._get_available_moves(),
        }

        return observation, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """Convert the current grid state to a numpy array."""
        obs = np.zeros((self.height, self.width), dtype=np.int32)
        grid = self.lot.grid()

        for y in range(self.height):
            for x in range(self.width):
                cell = grid[y][x]
                if cell == '#':
                    obs[y, x] = 1
                elif cell == '.':
                    obs[y, x] = 0
                else:
                    # Map vehicle IDs to numbers starting from 2
                    obs[y, x] = self.vehicle_ids.index(cell) + 2

        return obs

    def render(self, mode='human'):
        """Render the environment to the screen."""
        if mode == 'human':
            str(self.lot)
        return None

    def close(self):
        """Clean up environment resources."""
        pass