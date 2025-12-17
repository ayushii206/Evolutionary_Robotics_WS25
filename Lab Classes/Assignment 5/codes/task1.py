import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- Parameters ----------------
POP_SIZE = 40
GENERATIONS = 100
EVALS_PER_INDIVIDUAL = 1

ARENA_SIZE = 10.0
CHARGE_RADIUS = 1.5

BMAX = 1.0
DISCHARGE_RATE = 0.002

ROBOT_SPEED = 0.15
TIME_STEPS = 1000

MUTATION_STD = 0.1
NOISE_STD = 0.02

HIDDEN_NEURONS = 4

os.makedirs("results", exist_ok=True)

# ---------------- Helper Functions ----------------
def clip(x, lo=-1, hi=1):
    return np.maximum(lo, np.minimum(hi, x))

# ---------------- RNN Controller ----------------
class RNN:
    def __init__(self):
        self.W_in = np.random.randn(HIDDEN_NEURONS, 7)
        self.W_rec = np.random.randn(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.W_out = np.random.randn(2, HIDDEN_NEURONS)
        self.hidden = np.zeros(HIDDEN_NEURONS)

    def reset(self):
        self.hidden[:] = 0

    def step(self, x):
        self.hidden = np.tanh(self.W_in @ x + self.W_rec @ self.hidden)
        out = np.tanh(self.W_out @ self.hidden)
        return out

    def mutate(self):
        self.W_in += np.random.randn(*self.W_in.shape) * MUTATION_STD
        self.W_rec += np.random.randn(*self.W_rec.shape) * MUTATION_STD
        self.W_out += np.random.randn(*self.W_out.shape) * MUTATION_STD

    def clone(self):
        c = RNN()
        c.W_in = self.W_in.copy()
        c.W_rec = self.W_rec.copy()
        c.W_out = self.W_out.copy()
        return c

# ---------------- Environment ----------------
LAMP_POS = np.array([ARENA_SIZE / 2, ARENA_SIZE / 2])

def on_charging_area(pos):
    return np.linalg.norm(pos - LAMP_POS) < CHARGE_RADIUS

def light_intensity(pos):
    dist = np.linalg.norm(pos - LAMP_POS)
    return clip(1 / (1 + dist))

def get_sensors(pos, heading, battery):
    front = clip((ARENA_SIZE/2 - abs(pos[0])) / (ARENA_SIZE/2))
    left = clip((ARENA_SIZE/2 - abs(pos[1])) / (ARENA_SIZE/2))
    right = front
    lf = light_intensity(pos)
    lb = light_intensity(pos - np.array([np.cos(heading), np.sin(heading)]))
    ground = 0 if on_charging_area(pos) else 1
    battery_norm = battery / BMAX
    return np.array([front, left, right, lf, lb, ground, battery_norm])

# ---------------- Simulation ----------------
def simulate(controller, record=False):
    pos = np.random.uniform(-ARENA_SIZE/2, ARENA_SIZE/2, size=2)
    heading = np.random.uniform(0, 2*np.pi)
    battery = BMAX
    fitness = 0

    traj = np.zeros((TIME_STEPS, 2))
    batt_log = np.zeros(TIME_STEPS)

    controller.reset()

    for t in range(TIME_STEPS):
        sensors = get_sensors(pos, heading, battery)
        wheels = controller.step(sensors)
        wheels += np.random.randn(2) * NOISE_STD
        wheels = clip(wheels)

        v = np.mean(wheels)
        heading += (wheels[1] - wheels[0]) * 0.3
        pos += ROBOT_SPEED * np.array([np.cos(heading), np.sin(heading)]) * v

        battery -= DISCHARGE_RATE

        if on_charging_area(pos):
            battery = BMAX
            if record:
                traj[t] = pos
                batt_log[t] = battery
            continue

        i = np.mean(sensors[:3])
        fitness += abs(v) * (1 - i)

        if record:
            traj[t] = pos
            batt_log[t] = battery

        if battery <= 0:
            break

    return fitness, traj[:t+1], batt_log[:t+1]

# ---------------- Evolutionary Algorithm ----------------
population = [RNN() for _ in range(POP_SIZE)]

best_fitness = []
avg_fitness = []

best_controller = None
best_score = -np.inf

for gen in range(GENERATIONS):
    fitnesses = []

    for indiv in population:
        scores = []
        for _ in range(EVALS_PER_INDIVIDUAL):
            f, _, _ = simulate(indiv, record=False)
            scores.append(f)
        fitnesses.append(np.mean(scores))

    fitnesses = np.array(fitnesses)

    best_idx = np.argmax(fitnesses)
    if fitnesses[best_idx] > best_score:
        best_score = fitnesses[best_idx]
        best_controller = population[best_idx].clone()

    best_fitness.append(fitnesses.max())
    avg_fitness.append(fitnesses.mean())

    print(f"Gen {gen:03d} | Best: {fitnesses.max():.2f} | Avg: {fitnesses.mean():.2f}")

    elite = population[best_idx]
    new_pop = [elite.clone()]

    while len(new_pop) < POP_SIZE:
        parent = elite.clone()
        parent.mutate()
        new_pop.append(parent)

    population = new_pop

# ---------------- Results ----------------
output_folder = "Lab Classes/Assignment 5/output"  
os.makedirs(output_folder, exist_ok=True)

plt.figure()
plt.plot(best_fitness, label="Best")
plt.plot(avg_fitness, label="Average")
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Evolution")
plt.savefig(os.path.join(output_folder, "fitness.png"))

fitness, traj, battery = simulate(best_controller, record=True)

plt.figure()
plt.plot(traj[:,0], traj[:,1])
circle = plt.Circle(LAMP_POS, CHARGE_RADIUS, color='black', fill=False)
plt.gca().add_patch(circle)
plt.title("Robot Trajectory")
plt.axis("equal")
plt.savefig(os.path.join(output_folder, "trajectory.png"))

plt.figure()
plt.plot(battery)
plt.xlabel("Time")
plt.ylabel("Battery Level")
plt.title("Battery Over Time")
plt.savefig(os.path.join(output_folder, "battery.png"))

plt.show()
