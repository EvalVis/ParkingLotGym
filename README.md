# ParkingLotGym

[![codecov](https://codecov.io/gh/EvalVis/ParkingLotGym/branch/main/graph/badge.svg)](https://codecov.io/gh/EvalVis/ParkingLotGym)

[![PyPI version](https://badge.fury.io/py/ParkingLotGym.svg)](https://pypi.org/project/parkinglotgym/)

A custom AI Gymnasium environment for the Rush hour puzzle: https://en.wikipedia.org/wiki/Rush_Hour_(puzzle).

Library used: [![GitHub](https://img.shields.io/badge/GitHub-EvalVis/ParkingLot-black?style=flat&logo=github)](https://github.com/EvalVis/ParkingLot).

# Example solvable in 60 moves

Random moves are used for this demo. Click on .gif if still.

![ParkingLot60](images/parking_lot_60.gif)

# Example solvable in 5 moves

Random moves are used for this demo. Click if .gif if still.

![ParkingLot5](images/parking_lot_5.gif)

## Environment Details

- **Action Space**: MultiDiscrete([num_vehicles, max_moves*2], start=[2, -max_moves]) - First value selects vehicle ID (starting at 2), second value indicates steps to move (negative for backward, positive for forward). Moving 0 steps is not allowed.
- **Observation Space**: `Box(0, num_vehicles+1, (height, width), int32)`.
Contains values: `0` for empty cells, `1` for walls, `2` and up for vehicles. '2` represents the main vehicle.
- **Reward**: `0` if the puzzle is solved, `-1` if not solved yet, `-2` if invalid move.
- **Done**: `True` if the puzzle is solved, `False` otherwise.