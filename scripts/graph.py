import csv
import matplotlib.pyplot as plt

# /common/home/do309/CS562/walk-these-ways/runs/gait-conditioned-agility/2023-11-15/train/
path = "/common/home/do309/CS562/walk-these-ways/runs/gait-conditioned-agility/2023-11-14/train/"
checkpoint = "072139.652191"
file = "/success_rate.txt"

full_path = path + checkpoint + file

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
