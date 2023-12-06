import csv
import matplotlib.pyplot as plt

# path = "/common/home/ab2465/CS562/walk-these-ways/runs/gait-conditioned-agility/2023-11-18/train/"
# checkpoint = "162825.521162"
path = "/common/home/ab2465/Downloads"
file = "/success_rate_static.txt"

full_path = path + file

success_rate = []

with open(full_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header line
    for line in csv_reader:
        success_rate.append(float(line[0]))
    # for line in file:
    #    success_rate.append(float(line.strip().split(',')[0]))  # Convert to float

success_rate = success_rate[1:]

plt.plot(range(len(success_rate)), success_rate)
plt.xlabel("Iterations")
plt.ylabel("Success Rate")
plt.title("Success Rate Over Iterations")
plt.grid(True)

# Set y-axis ticks from 0 to max success rate with a step of 1
max_success_rate = max(success_rate)
plt.yticks(range(0, int(max_success_rate) + 1, 2))

plt.show()

import csv
import matplotlib.pyplot as plt

# path = "/common/home/ab2465/CS562/walk-these-ways/runs/gait-conditioned-agility/2023-11-18/train/"
# checkpoint = "162825.521162"
path = "/common/home/ab2465/Downloads"
file = "/success_rate_static.txt"

full_path = path + file

# Initialize lists for each metric
episode_reward = []
goal_distance_rew = []
wall_dist_rew = []
goal_rew = []

with open(full_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header line
    for line in csv_reader:
        # Assuming each line has the structure: 
        # Success Rate, Episode Reward, Goal Distance Rew, Wall Dist Rew, Goal Rew
        episode_reward.append(float(line[1]))
        goal_distance_rew.append(float(line[2]))
        wall_dist_rew.append(float(line[3]))
        goal_rew.append(float(line[4]))

# Create a range for the x-axis based on the number of entries
x_range = range(len(episode_reward))

# Plotting each metric
plt.plot(x_range, episode_reward, label="Episode Reward")
plt.plot(x_range, goal_distance_rew, label="Goal Distance Rew")
plt.plot(x_range, wall_dist_rew, label="Wall Dist Rew")
plt.plot(x_range, goal_rew, label="Goal Rew")

# Adding labels, title, and grid
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.title("Training Metrics Over Iterations")
plt.grid(True)

# Adding a legend to differentiate the lines
plt.legend()

# Display the plot
plt.show()
