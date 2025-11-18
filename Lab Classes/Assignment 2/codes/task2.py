import pygame
import numpy as np
import random
import math

# --- Simulation constants ---
WIDTH, HEIGHT = 800, 800
ROBOT_RADIUS = 15
SENSOR_RANGE = 120
DT = 0.1
EVAL_TIME = 25  # seconds per evaluation
GRID_SIZE = 10
BG_COLOR = (245, 245, 245)

# --- Hill climber constants ---
MUTATION_RATE = 0.3
MAX_GENERATIONS = 15
NUM_RANDOM_WALLS = 12  # number of random walls

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hill Climber Robot with Random Walls and Sliding Collisions")
clock = pygame.time.Clock()


# --- Environment setup ---
def generate_random_walls(num_walls):
    walls = [
        ((50, 50), (750, 50)),
        ((750, 50), (750, 750)),
        ((750, 750), (50, 750)),
        ((50, 750), (50, 50)),
    ]
    for _ in range(num_walls):
        x1, y1 = random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)
        x2 = x1 + random.randint(-250, 250)
        y2 = y1 + random.randint(-250, 250)
        x2 = np.clip(x2, 50, WIDTH - 50)
        y2 = np.clip(y2, 50, HEIGHT - 50)
        walls.append(((x1, y1), (x2, y2)))
    return walls


WALLS = generate_random_walls(NUM_RANDOM_WALLS)


def random_genome():
    return np.random.uniform(-2, 2, 6)


def mutate(genome):
    new_g = genome.copy()
    for i in range(len(new_g)):
        if random.random() < 0.5:
            new_g[i] += np.random.normal(0, MUTATION_RATE)
    return new_g


def line_point_distance(p, a, b):
    ap = np.array(p) - np.array(a)
    ab = np.array(b) - np.array(a)
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    closest = np.array(a) + t * ab
    return np.linalg.norm(np.array(p) - closest)


def check_collision(pos):
    for wall in WALLS:
        if line_point_distance(pos, wall[0], wall[1]) < ROBOT_RADIUS:
            return True
    return False


def raycast(pos, angle):
    min_dist = SENSOR_RANGE
    hit_point = (pos[0] + math.cos(angle) * SENSOR_RANGE, pos[1] + math.sin(angle) * SENSOR_RANGE)

    for wall in WALLS:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        denom = (x1 - x2) * math.sin(angle) - (y1 - y2) * math.cos(angle)
        if denom == 0:
            continue
        t = ((x1 - pos[0]) * math.sin(angle) - (y1 - pos[1]) * math.cos(angle)) / denom
        u = -((x1 - x2) * (y1 - pos[1]) - (y1 - y2) * (x1 - pos[0])) / denom
        if 0 <= t <= 1 and 0 <= u <= SENSOR_RANGE:
            if u < min_dist:
                min_dist = u
                hit_point = (pos[0] + math.cos(angle) * u, pos[1] + math.sin(angle) * u)
    return 1 - min_dist / SENSOR_RANGE, hit_point


def sense_walls(pos, angle):
    sensors = []
    ray_endpoints = []
    for da in [-0.5, 0, 0.5]:
        s_val, s_point = raycast(pos, angle + da)
        sensors.append(s_val)
        ray_endpoints.append(s_point)
    return np.array(sensors), ray_endpoints


def evaluate(genome, visualize=True):
    pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)
    angle = 0.0
    visited = set()
    time_steps = int(EVAL_TIME / DT)

    for step in range(time_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0

        sensors, rays = sense_walls(pos, angle)
        m0, c0, m1, c1, m2, c2 = genome

        vl = m0 * sensors[0] + c0
        vr = m1 * sensors[2] + c1 + m2 * sensors[1] + c2

        v = (vl + vr) / 2
        omega = (vr - vl) / (2 * ROBOT_RADIUS)

        # Proposed movement
        proposed_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * v * 5
        new_pos = pos.copy()

        # X-axis movement
        temp_pos = new_pos + np.array([proposed_pos[0] - pos[0], 0])
        if not check_collision(temp_pos):
            new_pos[0] = temp_pos[0]
        # Y-axis movement
        temp_pos = new_pos + np.array([0, proposed_pos[1] - pos[1]])
        if not check_collision(temp_pos):
            new_pos[1] = temp_pos[1]

        pos = new_pos
        angle += omega * DT

        # Keep inside arena
        pos[0] = np.clip(pos[0], 60, WIDTH - 60)
        pos[1] = np.clip(pos[1], 60, HEIGHT - 60)

        # Record visited cell
        cell = (int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE))
        visited.add(cell)

        if visualize:
            screen.fill(BG_COLOR)
            for w in WALLS:
                pygame.draw.line(screen, (0, 0, 0), w[0], w[1], 3)

            for (cx, cy) in visited:
                pygame.draw.rect(screen, (200, 255, 200),
                                 (cx * GRID_SIZE, cy * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            for (x, y) in rays:
                pygame.draw.line(screen, (255, 0, 0), pos.astype(int), (int(x), int(y)), 2)

            pygame.draw.circle(screen, (0, 0, 255), pos.astype(int), ROBOT_RADIUS)
            head_x = int(pos[0] + math.cos(angle) * ROBOT_RADIUS)
            head_y = int(pos[1] + math.sin(angle) * ROBOT_RADIUS)
            pygame.draw.line(screen, (255, 255, 0), pos.astype(int), (head_x, head_y), 3)

            pygame.display.flip()
            clock.tick(60)

    return len(visited)


def hill_climb():
    best_genome = random_genome()
    best_fitness = evaluate(best_genome, visualize=False)
    print(f"Initial fitness: {best_fitness:.2f}")

    for gen in range(MAX_GENERATIONS):
        new_genome = mutate(best_genome)
        new_fitness = evaluate(new_genome, visualize=False)
        if new_fitness > best_fitness:
            best_genome, best_fitness = new_genome, new_fitness
            print(f"✅ Gen {gen}: Improved fitness → {best_fitness:.2f}")
        else:
            print(f"Gen {gen}: No improvement ({new_fitness:.2f})")
    return best_genome, best_fitness


if __name__ == "__main__":
    print("🚀 Starting Hill Climber Robot Evolution...")
    best, fit = hill_climb()
    print("\n🎯 Evolution complete!")
    print(f"Best genome: {np.round(best, 3)}")
    print(f"Best fitness: {fit:.2f}")

    print("\n🎬 Running final simulation for best genome...")
    evaluate(best, visualize=True)
    pygame.quit()
