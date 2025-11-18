import random
import string
import matplotlib.pyplot as plt
import os

# --- Target String ---
TARGET = "charles darwin was always seasick"
CHAR_SET = string.ascii_lowercase + " "
LENGTH = len(TARGET)


def random_string():
    """Generate a random string of the same length as TARGET."""
    return ''.join(random.choice(CHAR_SET) for _ in range(LENGTH))


def fitness(candidate):
    """Fitness is the number of characters matching the target."""
    return sum(c1 == c2 for c1, c2 in zip(candidate, TARGET))


def mutate(candidate):
    """Mutate one random position by replacing with a random character."""
    i = random.randint(0, LENGTH - 1)
    new_char = random.choice(CHAR_SET)
    mutated = list(candidate)
    mutated[i] = new_char
    return ''.join(mutated)


def hill_climb(save_to_file=True, plot_graph=True):
    """Run the hill climbing algorithm with logging and plotting."""

    current = random_string()
    current_fitness = fitness(current)
    generation = 0
    fitness_history = [(generation, current_fitness)]

    print(f"Target: {TARGET}")
    print(f"Initial: {current}\n")

    generations_log = []
    generations_log.append((generation, current, current_fitness))

    while current != TARGET:
        generation += 1
        candidate = mutate(current)
        candidate_fitness = fitness(candidate)

        if candidate_fitness >= current_fitness:
            current, current_fitness = candidate, candidate_fitness

        fitness_history.append((generation, current_fitness))
        generations_log.append((generation, current, current_fitness))

        print(f"Generation {generation:5d}: {current}  | Fitness = {current_fitness}")

    print(f"\nReached target in {generation} generations.")

    # ---- Save output to a text file ----
    if save_to_file:
        with open("hill_climb_results.txt", "w") as f:
            f.write("Hill Climber Evolution Log\n")
            f.write(f"Target String: {TARGET}\n")
            f.write(f"Total Generations: {generation}\n\n")

            f.write("First 5 Generations:\n")
            for g in generations_log[:5]:
                f.write(f"Generation {g[0]}: {g[1]} | Fitness = {g[2]}\n")

            f.write("\nLast 5 Generations:\n")
            for g in generations_log[-5:]:
                f.write(f"Generation {g[0]}: {g[1]} | Fitness = {g[2]}\n")

            f.write(f"\nFinal Result Reached in {generation} Generations.\n")

    # ---- Plot and save fitness vs generation graph ----
    if plot_graph:
        gens = [g for g, _ in fitness_history]
        fits = [f for _, f in fitness_history]
        plt.figure(figsize=(8, 5))
        plt.plot(gens, fits, linewidth=2)
        plt.title("Fitness Progression over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Number of correct characters)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        output_folder = "Lab Classes/Assignment 2/output"  
        os.makedirs(output_folder, exist_ok=True)

        filename = "fitness_progression.jpg"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        plt.close()

    return generation


if __name__ == "__main__":
    # Run one simulation and record results
    gens = hill_climb()
    print(f"\nAll data saved to 'hill_climb_results.txt' and 'fitness_progression.jpg'.")
