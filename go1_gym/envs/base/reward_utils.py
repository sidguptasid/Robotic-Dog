from tkinter import Y
import numpy as np

def get_reward(position, orientation, velocities):
    """Reward function for the task"""
    y_pos = position[:, 1]
    x_pos = position[:, 0]

    success_indices = np.where(y_pos >= 1.5)
    # static_indices = np.where(np.abs(velocities) < 0.01)
    indices_to_update = np.where(y_pos < 1.5)

    rewards = np.zeros_like(y_pos)

    # rewards[static_indices] = -10
    rewards[success_indices] = 100

    y_dist = 1.5 - y_pos
    orientation_dist = np.abs(orientation)
    closest_wall_dist = min(
        np.abs(y_pos + 2),
        np.abs(x_pos - 1),
        np.abs(x_pos + 1),
    )

    closest_wall_dist[success_indices] = 0

    rewards[indices_to_update] = - y_dist[indices_to_update] - orientation_dist[indices_to_update] + closest_wall_dist[indices_to_update]

    return rewards

x_values = np.arange(-1, 1, 0.01)
y_values = np.arange(-2, 2, 0.01)

x, y = np.meshgrid(x_values, y_values)