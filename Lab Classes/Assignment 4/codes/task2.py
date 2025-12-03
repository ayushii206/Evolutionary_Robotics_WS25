import numpy as np
import random
import math
import os
import time
import matplotlib.pyplot as plt

try:
    import cma
    HAS_CMA = True
except Exception:
    HAS_CMA = False

WIDTH, HEIGHT = 800, 800
ROBOT_RADIUS = 15
SENSOR_RANGE = 120
DT = 0.1
EVAL_TIME = 5
GRID_SIZE = 10
BG_COLOR = (245, 245, 245)

USE_ANN = True
ANN_DIM = 14
LIN_DIM = 6

CMA_POP = 5
CMA_SIGMA = 0.6
CMA_MAX_ITERS = 10
FALLBACK_POP = 10
FALLBACK_GENS = 10

OUTPUT_DIR = "Lab Classes/Assignment 4/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VISUALIZE_FINAL = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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

NUM_RANDOM_WALLS = 12
WALLS = generate_random_walls(NUM_RANDOM_WALLS)

def line_point_distance(p, a, b):
    ap = np.array(p) - np.array(a)
    ab = np.array(b) - np.array(a)
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(np.array(p) - np.array(a))
    t = np.clip(np.dot(ap, ab) / denom, 0, 1)
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
    for da in [-0.5, 0.0, 0.5]:
        s_val, s_point = raycast(pos, angle + da)
        sensors.append(s_val)
        ray_endpoints.append(s_point)
    return np.array(sensors), ray_endpoints

def activation(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

class ANN:
    def __init__(self, flat):
        f = np.array(flat).flatten()
        assert f.size == ANN_DIM
        self.w1 = f[0:6].reshape((3, 2))
        self.b1 = f[6:8]
        self.w2 = f[8:12].reshape((2, 2))
        self.b2 = f[12:14]

    def forward(self, sensors):
        h = activation(np.dot(sensors, self.w1) + self.b1)
        o = activation(np.dot(h, self.w2) + self.b2)
        return o

def evaluate_genome_linear(genome, visualize=False, random_start=False):
    pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float) if not random_start else np.array(
        [random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)])
    angle = 0.0 if not random_start else random.uniform(0, 2 * math.pi)
    visited = set()
    steps = int(EVAL_TIME / DT)
    for step in range(steps):
        sensors, rays = sense_walls(pos, angle)
        m0, c0, m1, c1, m2, c2 = genome
        vl = m0 * sensors[0] + c0
        vr = m1 * sensors[2] + c1 + m2 * sensors[1] + c2
        v = (vl + vr) / 2.0
        omega = (vr - vl) / (2 * ROBOT_RADIUS)
        proposed_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * v * 5
        new_pos = pos.copy()
        temp_pos = new_pos + np.array([proposed_pos[0] - pos[0], 0])
        if not check_collision(temp_pos):
            new_pos[0] = temp_pos[0]
        temp_pos = new_pos + np.array([0, proposed_pos[1] - pos[1]])
        if not check_collision(temp_pos):
            new_pos[1] = temp_pos[1]
        pos = new_pos
        angle += omega * DT
        pos[0] = np.clip(pos[0], 60, WIDTH - 60)
        pos[1] = np.clip(pos[1], 60, HEIGHT - 60)
        cell = (int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE))
        visited.add(cell)
    return len(visited)

def evaluate_genome_ann(flat_weights, visualize=False, random_start=False):
    ann = ANN(flat_weights)
    pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float) if not random_start else np.array(
        [random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)])
    angle = 0.0 if not random_start else random.uniform(0, 2 * math.pi)
    visited = set()
    steps = int(EVAL_TIME / DT)
    trajectory = []
    for step in range(steps):
        sensors, rays = sense_walls(pos, angle)
        outs = ann.forward(sensors)
        vl = outs[0] * 1.0
        vr = outs[1] * 1.0
        v = (vl + vr) / 2.0
        omega = (vr - vl) / (2 * ROBOT_RADIUS)
        proposed_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * v * 5
        new_pos = pos.copy()
        temp_pos = new_pos + np.array([proposed_pos[0] - pos[0], 0])
        if not check_collision(temp_pos):
            new_pos[0] = temp_pos[0]
        temp_pos = new_pos + np.array([0, proposed_pos[1] - pos[1]])
        if not check_collision(temp_pos):
            new_pos[1] = temp_pos[1]
        pos = new_pos
        angle += omega * DT
        pos[0] = np.clip(pos[0], 60, WIDTH - 60)
        pos[1] = np.clip(pos[1], 60, HEIGHT - 60)
        cell = (int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE))
        visited.add(cell)
        trajectory.append((pos[0], pos[1]))
    return len(visited), trajectory

def run_cma_es(fitness_func, dim, popsize=CMA_POP, sigma=CMA_SIGMA, max_iters=CMA_MAX_ITERS, random_start=False):
    if HAS_CMA:
        opts = {'popsize': popsize}
        es = cma.CMAEvolutionStrategy(np.zeros(dim), sigma, opts)
        best_history = []
        avg_history = []
        for gen in range(max_iters):
            candidates = es.ask()
            vals = []
            for cand in candidates:
                v = fitness_func(cand, random_start=random_start)
                vals.append(-v)
            es.tell(candidates, vals)
            fevals = -np.array(vals)
            best_history.append(np.max(fevals))
            avg_history.append(np.mean(fevals))
            print(f"CMA gen {gen:3d} best {best_history[-1]:.1f} avg {avg_history[-1]:.1f}")
        best = es.result.xbest
        return best, best_history, avg_history
    else:
        print("cma not installed: using fallback EA (slower). Install 'cma' for better performance.")
        pop = [np.random.randn(dim) for _ in range(FALLBACK_POP)]
        best_history = []
        avg_history = []
        for g in range(FALLBACK_GENS):
            fitnesses = []
            for ind in pop:
                fitnesses.append(fitness_func(ind, random_start=random_start))
            fitnesses = np.array(fitnesses)
            best_idx = np.argmax(fitnesses)
            best_history.append(fitnesses[best_idx])
            avg_history.append(np.mean(fitnesses))
            selected = [pop[i] for i in fitnesses.argsort()[-(FALLBACK_POP // 4):]]
            newpop = selected.copy()
            while len(newpop) < FALLBACK_POP:
                p1, p2 = random.sample(selected, 2)
                child = np.where(np.random.rand(dim) < 0.5, p1, p2) + np.random.randn(dim) * 0.2
                newpop.append(child)
            pop = newpop
            print(f"EA gen {g:3d} best {best_history[-1]:.1f} avg {avg_history[-1]:.1f}")
        fitnesses = [fitness_func(ind) for ind in pop]
        return pop[np.argmax(fitnesses)], best_history, avg_history

def run_experiment(use_ann=True, nondet=False):
    if use_ann:
        dim = ANN_DIM
        fitness_f = lambda w, random_start=False: evaluate_genome_ann(w, random_start=random_start)[0]
    else:
        dim = LIN_DIM
        fitness_f = lambda w, random_start=False: evaluate_genome_linear(w, random_start=random_start)

    print(f"Starting {'ANN' if use_ann else 'Linear'} experiment - {'random starts' if nondet else 'fixed start'}")
    best, best_hist, avg_hist = run_cma_es(fitness_f, dim, random_start=nondet)
    plt.figure(figsize=(8,4))
    plt.plot(best_hist, label='best')
    plt.plot(avg_hist, label='avg')
    plt.title(f"{'ANN' if use_ann else 'Linear'} - {'Random' if nondet else 'Deterministic'} evolution")
    plt.xlabel("Generation")
    plt.ylabel("Visited cells")
    plt.legend()
    fname_fit = os.path.join(OUTPUT_DIR, f"{'ann' if use_ann else 'lin'}_{'nondet' if nondet else 'det'}_fitness.png")
    plt.tight_layout(); plt.savefig(fname_fit); plt.close()
    print("Saved fitness plot to", fname_fit)
    if use_ann:
        visited, traj = evaluate_genome_ann(best, visualize=False, random_start=nondet)
    else:
        visited = evaluate_genome_linear(best, visualize=False, random_start=nondet)
        traj = []
        pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float) if not nondet else np.array([random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)])
        angle = 0.0 if not nondet else random.uniform(0, 2 * math.pi)
        for step in range(int(EVAL_TIME / DT)):
            sensors, _ = sense_walls(pos, angle)
            m0, c0, m1, c1, m2, c2 = best
            vl = m0 * sensors[0] + c0
            vr = m1 * sensors[2] + c1 + m2 * sensors[1] + c2
            v = (vl + vr) / 2.0
            omega = (vr - vl) / (2 * ROBOT_RADIUS)
            proposed_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * v * 5
            new_pos = pos.copy()
            temp_pos = new_pos + np.array([proposed_pos[0] - pos[0], 0])
            if not check_collision(temp_pos): new_pos[0] = temp_pos[0]
            temp_pos = new_pos + np.array([0, proposed_pos[1] - pos[1]])
            if not check_collision(temp_pos): new_pos[1] = temp_pos[1]
            pos = new_pos
            angle += omega * DT
            pos[0] = np.clip(pos[0], 60, WIDTH - 60)
            pos[1] = np.clip(pos[1], 60, HEIGHT - 60)
            traj.append((pos[0], pos[1]))
    if len(traj) > 0:
        xs, ys = zip(*traj)
        plt.figure(figsize=(6,6))
        plt.plot(xs, ys, linewidth=1)
        plt.title(f"{'ANN' if use_ann else 'Linear'} {'random' if nondet else 'det'} trajectory (visited={visited})")
        plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
        plt.gca().set_aspect('equal')
        fname_traj = os.path.join(OUTPUT_DIR, f"{'ann' if use_ann else 'lin'}_{'nondet' if nondet else 'det'}_traj.png")
        plt.tight_layout(); plt.savefig(fname_traj); plt.close()
        print("Saved trajectory plot to", fname_traj)
    np.save(os.path.join(OUTPUT_DIR, f"best_genome_{'ann' if use_ann else 'lin'}_{'nondet' if nondet else 'det'}.npy"), best)
    print("Saved best genome.")
    return best, best_hist, avg_hist

def visualize_best(genome, use_ann=True, random_start=False):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Best agent visualization")
    clock = pygame.time.Clock()
    running = True
    if use_ann:
        visited, traj = evaluate_genome_ann(genome, visualize=False, random_start=random_start)
    else:
        visited = evaluate_genome_linear(genome, visualize=False, random_start=random_start)
        traj = []
        pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float) if not random_start else np.array([random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)])
        angle = 0.0 if not random_start else random.uniform(0, 2 * math.pi)
        for step in range(int(EVAL_TIME / DT)):
            sensors, rays = sense_walls(pos, angle)
            m0, c0, m1, c1, m2, c2 = genome
            vl = m0 * sensors[0] + c0
            vr = m1 * sensors[2] + c1 + m2 * sensors[1] + c2
            v = (vl + vr) / 2.0
            omega = (vr - vl) / (2 * ROBOT_RADIUS)
            proposed_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * v * 5
            new_pos = pos.copy()
            temp_pos = new_pos + np.array([proposed_pos[0] - pos[0], 0])
            if not check_collision(temp_pos): new_pos[0] = temp_pos[0]
            temp_pos = new_pos + np.array([0, proposed_pos[1] - pos[1]])
            if not check_collision(temp_pos): new_pos[1] = temp_pos[1]
            pos = new_pos
            angle += omega * DT
            pos[0] = np.clip(pos[0], 60, WIDTH - 60)
            pos[1] = np.clip(pos[1], 60, HEIGHT - 60)
            traj.append((pos[0], pos[1]))
    tstep = 0
    visited_cells = set()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(BG_COLOR)
        for w in WALLS:
            pygame.draw.line(screen, (0, 0, 0), w[0], w[1], 3)
        draw_until = min(tstep, len(traj)-1)
        for i in range(draw_until+1):
            x, y = int(traj[i][0]), int(traj[i][1])
            cell = (int(x // GRID_SIZE), int(y // GRID_SIZE))
            visited_cells.add(cell)
        for (cx, cy) in visited_cells:
            pygame.draw.rect(screen, (200, 255, 200), (cx * GRID_SIZE, cy * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        if draw_until >= 0:
            pts = traj[:draw_until+1]
            if len(pts) > 1:
                pygame.draw.lines(screen, (255, 165, 0), False, pts, 2)
            x, y = int(pts[-1][0]), int(pts[-1][1])
            pygame.draw.circle(screen, (0, 0, 255), (x, y), ROBOT_RADIUS)
        pygame.display.flip()
        tstep += 1
        if tstep > len(traj) + 50:
            time.sleep(1.0)
            running = False
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    best_det, bh_det, ah_det = run_experiment(use_ann=USE_ANN, nondet=False)
    best_nd, bh_nd, ah_nd = run_experiment(use_ann=USE_ANN, nondet=True)
    if VISUALIZE_FINAL:
        visualize_best(best_det, use_ann=USE_ANN, random_start=False)
        visualize_best(best_nd, use_ann=USE_ANN, random_start=True)
    print("All done. Outputs are in the 'outputs' folder.")
