import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

# Ackley objective function in 3D
def ackley(x):
    x = np.asarray(x)
    assert x.shape[-1] == 3
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    sum_sq = np.sum(x**2, axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / 3.0))
    term2 = -np.exp(np.sum(np.cos(c * x), axis=-1) / 3.0)
    return term1 + term2 + a + np.e

def fitness_from_ackley(pop):
    f = ackley(pop)
    # transform into maximization fitness (higher = better)
    return 1.0 / (f + 1.0)

# EA Operators: selection, crossover, mutation
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

def mutate(ind, mutation_rate=0.1, sigma=0.5, lower=-32.768, upper=32.768):
    mask = rng.random(ind.shape) < mutation_rate
    ind2 = ind.copy()
    ind2[mask] += rng.normal(0, sigma, size=mask.sum())
    np.clip(ind2, lower, upper, out=ind2)
    return ind2

def run_ackley_ea(pop_size=100, generations=200, mutation_rate=0.12, sigma=1.0,
                  crossover_prob=0.9, elitism=1, tournament_k=3):
    lower, upper = -32.768, 32.768
    pop = rng.uniform(lower, upper, size=(pop_size, 3))
    best_history = []
    avg_history = []
    for g in range(generations):
        fitnesses = fitness_from_ackley(pop)
        best_idx = np.argmax(fitnesses)
        best_history.append(fitnesses[best_idx])
        avg_history.append(np.mean(fitnesses))
        new_pop = []
        if elitism > 0:
            elites = np.argsort(fitnesses)[-elitism:][::-1]
            for e in elites:
                new_pop.append(pop[e].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitnesses, k=tournament_k)
            p2 = tournament_select(pop, fitnesses, k=tournament_k)
            if rng.random() < crossover_prob:
                child = arithmetic_crossover(p1, p2)
            else:
                child = p1.copy()
            child = mutate(child, mutation_rate=mutation_rate, sigma=sigma, lower=lower, upper=upper)
            new_pop.append(child)
        pop = np.vstack(new_pop)[:pop_size]
    fitnesses = fitness_from_ackley(pop)
    best_idx = np.argmax(fitnesses)
    return {
        "best_individual": pop[best_idx],
        "best_fitness": fitnesses[best_idx],
        "best_history": np.array(best_history),
        "avg_history": np.array(avg_history)
    }

if __name__ == "__main__":
    POP = 80
    GENS = 150
    MUT = 0.15
    SIG = 1.0
    ELIT = 2

    print("Running Task 1 EA on 3D Ackley")
    res = run_ackley_ea(pop_size=POP, generations=GENS, mutation_rate=MUT,
                        sigma=SIG, elitism=ELIT, tournament_k=3)

    best = res["best_individual"]
    best_f = res["best_fitness"]
    print("Best individual (x,y,z):", best)
    print("Best fitness (1/(f+1)):", best_f)
    print("Ackley f(x):", ackley(best))

    output_folder = "Lab Classes/Assignment 3/output"  
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(res["best_history"], label="best")
    plt.plot(res["avg_history"], label="avg")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/(f+1))")
    plt.title("Task 1: Fitness over generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename1 = "task1_best.png"
    filepath1 = os.path.join(output_folder, filename1)
    plt.savefig(filepath1)
    print("Saved plot: task1_best.png")

    plt.figure(figsize=(6,4))
    plt.plot(res["avg_history"])
    plt.xlabel("Generation")
    plt.ylabel("Average fitness")
    plt.title("Task 1: Average fitness")
    plt.grid(True)
    plt.tight_layout()
    filename2 = "task1_avg.png"
    filepath2 = os.path.join(output_folder, filename2)
    plt.savefig(filepath2)
    print("Saved plot: task1_avg.png")

    # mutation rate effect for short runs
    mut_rates = [0.01, 0.05, 0.15, 0.3]
    print("\nParameter study: mutation rates (short runs)")
    for mr in mut_rates:
        r = run_ackley_ea(pop_size=50, generations=60, mutation_rate=mr, sigma=1.0, elitism=1)
        print(f" mutation_rate={mr:.3f} -> final best fitness = {r['best_fitness']:.6f}")

    print("\nTask 1 complete. Include 'task1_best.png' and 'task1_avg.png' in your submission.")
