# Evolutionary Robotics – Task Sheet 4  
**Winter Semester 2025**  
**Prof. Dr. Javad Ghofrani**  
**Deadline:** 04.12.2025  

---

# Overview

This assignment consists of two experiments:

1. **Task 1 — Optimization of a nonlinear test function (Ackley)**  
2. **Task 2 — Robot exploration using an EA-evolved Artificial Neural Network (ANN)**  
   - Deterministic start  
   - Non-deterministic start  
   - Analysis & visualization  
   - Optional improvements

---

# Task 1 — Optimization of the Ackley Function

## Objective  
Minimize the 3D Ackley function by evolving a vector:  

```
[x, y, z]
```

Fitness:

```
fitness = 1 / (ackley(x,y,z) + 1)
```

---

## Implementation

### EA representation
- Real vector in ℝ³  
- Range: [-32.768, 32.768]

### EA settings  
- Tournament selection (k=3)  
- Arithmetic crossover (p=0.9)  
- Gaussian mutation  
- Mutation rate study  
- Elitism = 2  
- Generations = 150  
- Population = 100  

---

## Results

- `task1_best.png`  
- `task1_avg.png`  

Both curves converge toward ≈1.0, the global optimum.

### Mutation rate summary:

| Mutation Rate | Final Best Fitness |
|---------------|-------------------|
| 0.01 | 0.919877 |
| 0.05 | 0.967569 |
| 0.15 | 0.999055 |
| 0.30 | 0.990688 |

Optimal range: **0.10–0.15**

---

# Task 2 — Robot Exploration with EA‑Evolved ANN (Partially done)#

## Objective  
Evolve an ANN controller that maximizes **arena coverage** (unique grid cells visited).

---

## Simulation Environment  
(using the provided Tutorial 2 pygame simulator)

- 800×800 px arena  
- 12 internal random walls  
- 3 ray sensors  
- Robot radius = 15 px  
- Grid = 10 px  
- Evaluation time = 20 s  
- DT = 0.1 s  

---

## ANN Controller  

- Inputs: 3 proximity sensors  
- Hidden: 2 neurons (tanh)  
- Output: 2 wheel commands  
- Total weights: 14  

Genome:

```
[w1 … w14]
```

---

## Evolution: CMA‑ES  

- Population = 20  
- σ = 0.6  
- Iterations = 80  
- Deterministic and random-start runs performed  

Fallback EA is used if CMA is unavailable.

---

# Results

### Deterministic (fixed start)
- `ann_det_fitness.png`  
- `ann_det_traj.png`  
- `best_genome_ann_det.npy`

### Non‑deterministic (random start)
- `ann_nondet_fitness.png`  
- `ann_nondet_traj.png`  
- `best_genome_ann_nondet.npy`

Both show increasing performance and improved exploration.

---

# Visualization

Two pygame visualizations:

1. Best deterministic agent  
2. Best random-start agent  

Includes walls, trajectory, sensor rays, and visited grid cells.

---

# Conclusion

- EA successfully optimized the Ackley function.  
- CMA‑ES efficiently evolved ANN controllers for robust exploration.  
- ANN agents generalize well across random starting conditions.

## 👥 Contributors:

- [Ayushi Arora](https://github.com/ayushii206)

## Acknowledgements:

- [Prof. Dr. Javad Ghofrani](https://www.h-brs.de/de/inf/prof-dr-javad-ghofrani)

- [Youssef Mahmoud Youssef](https://www.h-brs.de/de/inf/youssef-mahmoud-youssef)

- [Hochschule Bonn-Rhein-Sieg](https://www.h-brs.de/de)
