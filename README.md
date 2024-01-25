# Walk-these-ways (CS562: Advanced Robotics)

## Introduction

This project, part of CS562: Advanced Robotics, is an endeavor to train and demonstrate a robotic dog's ability to navigate and avoid obstacles. It leverages the Isaac Gym Simulator and the Unitree Go1 robot, focusing on achieving precise locomotion and navigation in a virtual environment. The project's ultimate goal is to enable the robot dog to move from point A to point B while avoiding collisions, simulating real-world scenarios where such technology could be vital in search and rescue, agriculture, logistics, and healthcare.

## Phase 1: Robot Locomotion

### Overview
Phase 1 aims to train a velocity-conditioned neural network policy. This policy will enable the robot to walk at commanded longitudinal, lateral, and angular velocities. The sensory data includes joint positions, velocities, and orientation of the gravity vector, which are essential for the robot's stable movement.

### Implementation
The phase begins with setting up the Isaac Gym simulator and then cloning the 'walk-these-ways' repository. The main task is to train the robot to follow commanded velocities within a specified range, familiarizing participants with the simulator, code base, and the robot's functionality.

### Results


[![Thumbnail for Plan] <img width="649" alt="Thumbnail for Plan" src="https://github.com/sidguptasid/walk-these-ways/assets/132853979/425aac2b-b737-4eb9-988c-9eebe89ee1bc">](https://github.com/sidguptasid/walk-these-ways/assets/132853979/425aac2b-b737-4eb9-988c-9eebe89ee1bc)

## Phase 2: Robot Navigation within Walls

### Overview
Building on Phase 1, Phase 2 focuses on navigating a walled corridor without colliding. This involves training another policy using reinforcement learning, Actor-Critic Proximal Policy algorithm (PPO), which will provide velocity commands to the locomotion policy. The robot dog can have a fixed as well as random initialization.

### Implementation
The task involves setting up an environment with walls, implementing a Navigator class, and employing the Actor-Critic Proximal Policy algorithm for training. The challenge is to navigate the robot from a start to a goal position, varying the initial position and orientation of the robot.


### Results







#### Dog walking towards the goal line (Fixed Initialization)
[![Thumbnail for Plan] <img width="649" alt="Thumbnail for 80" src="https://github.com/sidguptasid/walk-these-ways/assets/132853979/8fed88e4-1afd-4b2a-bdb6-79dfbd03eccc">](https://github.com/sidguptasid/walk-these-ways/assets/132853979/8fed88e4-1afd-4b2a-bdb6-79dfbd03eccc)

#### Dog walking towards the goal line (Random Initialization)
[![Thumbnail for Plan] <img width="649" alt="Thumbnail for 80" src="https://github.com/sidguptasid/walk-these-ways/assets/132853979/2dc4f819-21af-48f6-9cb7-752c0891fef9">](https://github.com/sidguptasid/walk-these-ways/assets/132853979/2dc4f819-21af-48f6-9cb7-752c0891fef9)

#### Dog walking towards the goal line (Random Initialization)
[![Thumbnail for Plan] <img width="649" alt="Thumbnail for 80" src="https://github.com/sidguptasid/walk-these-ways/assets/132853979/3ef63dde-eeb9-4f46-81a8-128f67f66034">](https://github.com/sidguptasid/walk-these-ways/assets/132853979/3ef63dde-eeb9-4f46-81a8-128f67f66034)




## Phase 3: Robot Locomotion with Obstacle Avoidance (In progress)

### Overview
The final phase introduces obstacle avoidance to the navigation policy. The policy now needs to consider the obstacle's location and size in the environment.

### Implementation
We add an immovable obstacle to the existing environment and train the policy to avoid it. The input space should include the obstacle's location and size.


#### Diagram of the environment with the obstacle
<img width="516" alt="Room" src="https://github.com/sidguptasid/walk-these-ways/assets/132853979/e24f2a9e-3746-4e48-83cc-3e660505039d">
### Results




https://github.com/sidguptasid/walk-these-ways/assets/132853979/fbf8e07c-d012-4234-8244-b28362e64345



https://github.com/sidguptasid/walk-these-ways/assets/132853979/461bc5c3-31e4-445c-8c45-6836a54497f5



https://github.com/sidguptasid/walk-these-ways/assets/132853979/e8934703-8e3d-438f-8dde-a8257ea397d0
