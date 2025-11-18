import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "task_sheet", "data")

print("Looking for data at:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

data = np.loadtxt(DATA_PATH)
labels = data[:, 1].astype(int)
X = data[:, 2:4]

# ANN and activation function

def phi(net):
    return 2.0 / (1.0 + np.exp(-2.0 * net)) - 1.0

def ann_output(weights, inputs):
    # weights = [w0 (bias), w1 (x), w2 (y)]
    net = weights[0] + weights[1] * inputs[:, 0] + weights[2] * inputs[:, 1]
    return phi(net)

def fitness_ann(weights):
    outs = ann_output(weights, X)
    preds = (outs > 0).astype(int)
    return (preds == labels).sum()

# EA operators: selection, crossover, mutation

def tournament_select(pop, fitnesses, k=3):
    idxs = rng.integers(0, pop.shape[0], size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return pop[best].copy()

def arithmetic_crossover(p1, p2):
    alpha = rng.random(p1.shape)
    return alpha * p1 + (1 - alpha) * p2

def mutate(ind, mutation_rate=0.15, sigma=0.5, lower=-10, upper=10):
    mask = rng.random(ind.shape) < mutation_rate
    ind2 = ind.copy()
    ind2[mask] += rng.normal(0, sigma, size=mask.sum())
    np.clip(ind2, lower, upper, out=ind2)
    return ind2

# Evolutionary Algorithm for ANN weights

def run_ann_ea(pop_size=80, generations=150, mutation_rate=0.15, sigma=0.7,
               crossover_prob=0.9, elitism=1, tournament_k=3):

    lower, upper = -10, 10
    pop = rng.uniform(lower, upper, size=(pop_size, 3))

    best_history = []
    avg_history = []

    for g in range(generations):
        fitnesses = np.array([fitness_ann(ind) for ind in pop])
        best_history.append(np.max(fitnesses))
        avg_history.append(np.mean(fitnesses))

        new_pop = []

        # Elitism
        elites = np.argsort(fitnesses)[-elitism:][::-1]
        for e in elites:
            new_pop.append(pop[e].copy())

        # Fill the rest of the population
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitnesses, k=tournament_k)
            p2 = tournament_select(pop, fitnesses, k=tournament_k)

            if rng.random() < crossover_prob:
                child = arithmetic_crossover(p1, p2)
            else:
                child = p1.copy()

            child = mutate(child, mutation_rate=mutation_rate, sigma=sigma)
            new_pop.append(child)

        pop = np.vstack(new_pop)[:pop_size]

    # Final results
    fitnesses = np.array([fitness_ann(ind) for ind in pop])
    best_idx = np.argmax(fitnesses)

    return {
        "best_individual": pop[best_idx],
        "best_fitness": int(fitnesses[best_idx]),
        "best_history": np.array(best_history),
        "avg_history": np.array(avg_history)
    }

if __name__ == "__main__":
    print("Running Task 2 EA...")

    res = run_ann_ea()

    w0, w1, w2 = res["best_individual"]
    score = res["best_fitness"]

    print("\nBest ANN weights:")
    print(" w0 (bias)   =", w0)
    print(" w1 (x)      =", w1)
    print(" w2 (y)      =", w2)
    print("\nCorrect classifications:", score, "/", len(labels))

    # Plots

    output_folder = "Lab Classes/Assignment 3/output"  
    os.makedirs(output_folder, exist_ok=True)
    
    plt.figure()
    plt.plot(res["best_history"])
    plt.title("Task 2: Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Correct classifications")
    plt.grid(True)
    filename1 = "task2_best.png"
    filepath1 = os.path.join(output_folder, filename1)
    plt.savefig(filepath1)

    plt.figure()
    plt.plot(res["avg_history"])
    plt.title("Task 2: Average Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Correct classifications")
    plt.grid(True)
    filename2 = "task2_avg.png"
    filepath2 = os.path.join(output_folder, filename2)
    plt.savefig(filepath2)

    # Plot data points and separating line
    
    xs = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 300)
    if abs(w2) < 1e-6:
        ys = np.full_like(xs, X[:,1].mean())
    else:
        ys = (w0 / w2) - (w1 / w2) * xs

    plt.figure(figsize=(6,5))
    plt.scatter(X[labels==0,0], X[labels==0,1], label="Class 0")
    plt.scatter(X[labels==1,0], X[labels==1,1], label="Class 1")
    plt.plot(xs, ys, color="black", label="Decision boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Task 2: Data & ANN Separating Line")
    plt.legend()
    plt.grid(True)
    filename3 = "task2_separating_line.png"
    filepath3 = os.path.join(output_folder, filename3)
    plt.savefig(filepath3)
    
    print("\nSaved: task2_best.png, task2_avg.png, task2_separating_line.png")
