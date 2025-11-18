# Evolutionary Robotics – Task Sheet 2  
**Winter Semester 2025**  
**Prof. Dr. Javad Ghofrani**  
**Deadline:** 05.11.2025 23:59  

---

## Objectives
- Implement a simple **Hill Climber** algorithm.  
- Understand its strengths and limitations depending on the fitness landscape.  
- Apply the hill climber first to an abstract text-matching task and then to evolve a simple robot behavior.  

---

## Task 1 – Hill Climber on a Target Sentence

### Overview
We evolve a string toward the target  
> `"charles darwin was always seasick"`  

The algorithm starts from a random string of equal length and mutates one random character per generation, accepting only if fitness (number of correct characters) does not decrease.

### Implementation Highlights
- Single-individual hill climber (no population).  
- Fitness = number of correct characters (33 max).  
- Accept new string if fitness ≥ previous.  
- Logs every generation with string + fitness.  

### Outputs
- `hill_climb_results.txt` → typical run log (generation & fitness).  
- `fitness_progression.jpg` → fitness vs generation plot.  
- `task1_b.md` → explanation why the problem is “good-natured”.  
- `task1_3.md` → mathematical estimation of expected generations (analytic vs empirical).  

### Results Summary
- Target reached after ≈ 5700 generations (expected ≈ 3600 ).  
- Monotonic fitness curve showing step-wise improvement.  
- Demonstrates a smooth, unimodal fitness landscape.

---

## Task 2 – Hill Climber for Robot Behavior

### Overview
A simple **reactive robot** evolves exploratory behavior inside a bounded 800×800 px arena with randomly generated walls.  
The controller genome encodes three linear mappings from proximity sensor inputs to wheel speeds:

\[
v_l = m_0 s_l + c_0, \quad
v_r = m_1 s_r + c_1 + m_2 s_m + c_2
\]

### Implementation Highlights
- Deterministic evaluation (fixed random seed = 42).  
- Fitness = number of unique grid cells visited during simulation.  
- Mutation = Gaussian perturbation of each parameter.  
- Acceptance rule: accept if fitness ≥ current (best).  
- Movement integrated with DT and clamped wheel speeds.  
- Visualization via Pygame and plots via Matplotlib.

### Outputs
Saved automatically to  
`Lab Classes/Assignment 2/output/`  

| File | Description |
|------|--------------|
| `fitness_log.txt` | Hill-climber generation log |
| `best_genome.txt` | Parameters of best controller |
| `trajectory_best.png` | Path of best controller in arena |
| `fitness_progression.png` | Best fitness vs generation |
| `walls.txt` | Wall layout for reproducibility |

### Results Summary
- Best fitness ≈ 2300–2600 cells visited (depending on seed).  
- Robot develops smooth wall-following/exploration behavior.  
- Further improvements possible by tuning mutation rate, evaluation time, or adding noise for robustness.

---

## Usage Instructions

### Run Task 1
```bash
python task1.py
```

#### Outputs: console log and fitness_progression.jpg

### Run Task 2
```bash
python task2.py
```