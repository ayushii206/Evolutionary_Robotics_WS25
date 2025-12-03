
import numpy as np
import matplotlib.pyplot as plt
import time
import os

np.random.seed(0)

POP = 1000           
MAX_GEN = 2000
DIM = 9 
ELITISM = 10
TOURNAMENT = 5
MUT_STD = 0.1
TARGET_FIT = 0.9

def activation(x):
    return 2.0 / (1.0 + np.exp(-2.0*x)) - 1.0

class ANN:
    def __init__(self, flat):
        # flat: length DIM
        arr = np.array(flat).flatten()
        assert arr.size == DIM
        self.w1 = arr[0:4].reshape((2,2))   
        self.b1 = arr[4:6]                
        self.w2 = arr[6:8].reshape((2,1))  
        self.b2 = arr[8]                   

    def forward(self, x):
        # x: shape (2,) or (n,2)
        x = np.asarray(x)
        single = (x.ndim == 1)
        if single:
            x = x.reshape(1,2)
        h = activation(np.dot(x, self.w1) + self.b1)
        o = activation(np.dot(h, self.w2) + self.b2)
        out = o.reshape(-1)
        return out[0] if single else out

inputs = [(0,0),(1,0),(0,1),(1,1)]

def xor_fitness(flat):
    net = ANN(flat)
    score = 0.0
    for a,b in inputs:
        target = a ^ b  
        out = net.forward(np.array([a,b]))  # in [-1,1]
        score += 1.0 - abs(target - out)
    return score / 4.0

def tournament_selection(pop, fits, k=TOURNAMENT):
    i = np.random.randint(len(pop))
    best = i
    for _ in range(k-1):
        j = np.random.randint(len(pop))
        if fits[j] > fits[best]:
            best = j
    return pop[best]

def uniform_crossover(p1, p2):
    mask = np.random.rand(p1.size) < 0.5
    child = np.where(mask, p1, p2)
    return child

def mutate(x, std=MUT_STD):
    return x + np.random.normal(0, std, size=x.shape)

population = np.random.uniform(-1,1,(POP,DIM))

best_history = []
avg_history = []
start_time = time.time()

best_solution = None
best_fit_val = -np.inf

for gen in range(1, MAX_GEN+1):
    
    fits = np.array([xor_fitness(ind) for ind in population])
    best_idx = np.argmax(fits)
    gen_best = fits[best_idx]
    gen_avg = fits.mean()
    best_history.append(gen_best)
    avg_history.append(gen_avg)

    if gen_best > best_fit_val:
        best_fit_val = gen_best
        best_solution = population[best_idx].copy()

    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:4d} | best {gen_best:.6f} | avg {gen_avg:.6f}")

    if gen_best >= TARGET_FIT:
        print(f"Target reached at generation {gen} with fitness {gen_best:.6f}")
        best_solution = population[best_idx].copy()
        break

    new_pop = []
    
    elite_idx = fits.argsort()[-ELITISM:]
    for i in elite_idx:
        new_pop.append(population[i].copy())

    
    while len(new_pop) < POP:
        p1 = tournament_selection(population, fits)
        p2 = tournament_selection(population, fits)
        child = uniform_crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)

    population = np.array(new_pop)

elapsed = time.time() - start_time
print(f"Done. Best fitness {best_fit_val:.6f}. Time: {elapsed:.1f}s")

output_folder = "Lab Classes/Assignment 4/output"  
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(best_history, label="best")
plt.plot(avg_history, label="average")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.title("Fitness over generations (XOR EA)")
plt.grid(True)
plt.tight_layout()
filename1 = "task1_fitness_curve.png"
filepath1 = os.path.join(output_folder, filename1)
plt.savefig(filepath1)
print("Saved fitness_curve.png")

best_ann = ANN(best_solution)
N = 100
xs = np.linspace(0,1,N)
ys = np.linspace(0,1,N)
Z = np.zeros((N,N))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        Z[j,i] = best_ann.forward(np.array([x,y]))  

plt.figure(figsize=(6,5))
plt.imshow(Z, extent=[0,1,0,1], origin='lower', interpolation='bilinear')
plt.colorbar(label='ANN output ([-1,1])')
plt.title("ANN output over input space [0,1]^2")
plt.xlabel("input 1")
plt.ylabel("input 2")
plt.tight_layout()
filename2 = "task1_xor_surface.png"
filepath2 = os.path.join(output_folder, filename2)
plt.savefig(filepath2)
print("Saved xor_surface.png")

print("Final ANN outputs for discrete XOR inputs:")
for a,b in inputs:
    out = best_ann.forward(np.array([a,b]))
    print(f"Input ({a},{b}) -> {out:.6f}")
