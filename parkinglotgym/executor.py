#!/usr/bin/env python3
"""
ParkingLot Executor - A single file implementation for executing multiple iterations
of the parking lot environment with rendering and sleep between moves.

This module contains:
- ParkingLotExecutor class for running multiple iterations
- Example usage functions
- All functionality in a single file
"""

import time
import random
from typing import Optional
from parking_lot_env import ParkingLotEnv


class ParkingLotExecutor:
    """Class to execute multiple iterations of the parking lot environment with rendering."""
    
    def __init__(self, moves: int = 60):
        """Initialize the executor with the specified number of moves for the Lot constructor.
        
        Args:
            moves: Number of moves to pass to the Lot constructor (default: 60)
        """
        self.moves = moves
        self.env = None
        
    def execute_iterations(self, num_iterations: int = 100, sleep_time: float = 0.1):
        """Execute the specified number of iterations with rendering and sleep.
        
        Args:
            num_iterations: Number of iterations to execute (default: 100)
            sleep_time: Time to sleep between moves in seconds (default: 0.1)
        """
        print(f"Starting {num_iterations} iterations with {sleep_time}s sleep between moves...")
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Create a new environment for each iteration
            self.env = ParkingLotEnv(self.moves)
            
            # Reset the environment
            observation, info = self.env.reset()
            
            # Render the initial state
            self.env.render()
            time.sleep(sleep_time)
            
            done = False
            step_count = 0
            max_steps = 200  # Prevent infinite loops
            
            while not done and step_count < max_steps:
                # Get available moves
                available_moves = info.get('available_moves', {})
                
                if not available_moves:
                    print("No available moves!")
                    break
                
                # Choose a random vehicle and move
                try:
                    # Get a random vehicle that has available moves
                    vehicles_with_moves = [vid for vid, moves in available_moves.items() if moves]
                    
                    if not vehicles_with_moves:
                        print("No vehicles can move!")
                        break
                    
                    # Select random vehicle and move
                    vehicle_id = random.choice(vehicles_with_moves)
                    vehicle_index = self.env.vehicle_ids.index(vehicle_id) + 2  # +2 because vehicles start at index 2
                    available_vehicle_moves = available_moves[vehicle_id]
                    move = random.choice(available_vehicle_moves)
                    
                    # Execute the move
                    observation, reward, done, truncated, info = self.env.step((vehicle_index, move))
                    
                    print(f"Step {step_count + 1}: Moved vehicle {vehicle_id} by {move} steps (reward: {reward})")
                    
                    # Render the new state
                    self.env.render()
                    time.sleep(sleep_time)
                    
                    step_count += 1
                    
                    if done:
                        print(f"Puzzle solved in {step_count} steps!")
                        time.sleep(sleep_time * 2)  # Extra pause for solved puzzle
                        break
                        
                except (ValueError, IndexError) as e:
                    print(f"Error during move: {e}")
                    break
            
            if step_count >= max_steps:
                print(f"Reached maximum steps ({max_steps}) without solving.")
            
            # Close the environment
            if self.env:
                self.env.close()
                
        print(f"\nCompleted all {num_iterations} iterations!")
    
    def close(self):
        """Clean up resources."""
        if self.env:
            self.env.close()


def run_demo():
    """Run a quick demo with 3 iterations for testing purposes."""
    print("ParkingLot Executor Demo")
    print("=" * 30)
    print("This demo will run 3 iterations with 0.5 second delays for visibility.")
    print("Press Ctrl+C to stop early if needed.\n")
    
    # Create executor with 60 moves (as specified in requirements)
    executor = ParkingLotExecutor(moves=60)
    
    try:
        # Execute just 3 iterations for demo purposes with longer sleep for visibility
        executor.execute_iterations(num_iterations=3, sleep_time=0.5)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources
        executor.close()
        print("Demo completed!")


def run_full_example():
    """Run the full example with 100 iterations as specified."""
    print("ParkingLot Executor Example")
    print("=" * 40)
    
    # Create executor with 60 moves (as specified)
    executor = ParkingLotExecutor(moves=60)
    
    try:
        # Execute 100 iterations with 0.1 second sleep between moves
        executor.execute_iterations(num_iterations=100, sleep_time=0.1)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources
        executor.close()
        print("Executor closed.")


if __name__ == "__main__":
    # When run directly, execute the full example
    run_full_example() 