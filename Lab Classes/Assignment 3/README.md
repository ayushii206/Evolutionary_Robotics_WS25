# Evolutionary Robotics – Task Sheet 3   
**Winter Semester 2025**  
**Prof. Dr. Javad Ghofrani**  
**Deadline:** 19.11.2025  

---

# Overview

This assignment contains two main parts:

1. **Task 1 — Classical optimization using an Evolutionary Algorithm (EA)**  
2. **Task 2 — Supervised classification using an EA-evolved Artificial Neural Network (ANN)**  

---

# Task 1 — Optimization of the Ackley Function

## Objective
Minimize the 3D Ackley function using an EA.  
Fitness used: `1 / (ackley(x,y,z) + 1)`

where ackley(x,y,z) is the Ackley function.

## Implementation

- Representation: 3D real vector  
```bash
[x, y, z]
```
- Initialization: Uniform random sampling from the range [−32.768,32.768]
- Tournament selection: k = 3  
- Arithmetic crossover probability: p=0.9
- Gaussian mutation per gene
- Mutation rate tested at several values
- Values clipped to the allowed range
- Generational replacement with elitism = 2
- Fixed number of generations (150) 

## Results

- Best fitness plot: Shows rapid improvement and convergence toward a fitness near 1.0, which corresponds to the Ackley global minimum.

- Average fitness plot: Shows rapid improvement and convergence toward a fitness near 1.0, which corresponds to the Ackley global minimum.

Both plots included:

- `task1_best.png`
- `task1_avg.png`

## Parameter Study

We tested the mutation rate parameter:

| Mutation Rate | Final best fitness | Notes |
| :--- | :---: | ---: |
| 0.01 | Lower | Not enough exploration |
| 0.05 | High | Good Convergence |
| 0.15 | Highest | Best Balance |
| 0.30 | Lower | Too noisy/ unstable |

Mutation rate Results:

```bash
mutation_rate=0.010 -> final best fitness = 0.919877
mutation_rate=0.050 -> final best fitness = 0.967569
mutation_rate=0.150 -> final best fitness = 0.999055
mutation_rate=0.300 -> final best fitness = 0.990688
```

Conclusion: **0.10–0.15 is optimal**.

---

# Task 2 — ANN Classification via EA

## Objective
Evolve a simple ANN with weights `[w0, w1, w2]` to classify 2D data.

## ANN
- Inputs: x, y  
- Bias: w0  
- Activation: tanh-like function  
- Classification: output < 0 → class 0, output > 0 → class 1  

## EA Setup
 
- Tournament selection: k=3  
- Arithmetic crossover: p = 0.9
- Gaussian mutation
- Weights constrained to [-10, 10]
- Generational replacement with elitism = 1 
- Population 80  
- Generations 150 

## Results

- Best fitness plot: Shows the improvement of correct classifications per generation.

- Average fitness plot: Shows the mean number of correct classifications.

- Seperating Line plot: The ANN’s decision boundary separates most points of different classes.

Plots included:
- `task2_best.png`
- `task2_avg.png`
- `task2_separating_line.png`

Best weights Results:

```bash
Best ANN weights:
 w0 (bias)   = -5.402619274083255
 w1 (x)      = 4.443060343310428
 w2 (y)      = 6.002909101054167

Correct classifications: 93 / 100
```

---

## 👥 Contributors:

- [Ayushi Arora](https://github.com/ayushii206)

## Acknowledgements:

- [Prof. Dr. Javad Ghofrani](https://www.h-brs.de/de/inf/prof-dr-javad-ghofrani)

- [Youssef Mahmoud Youssef](https://www.h-brs.de/de/inf/youssef-mahmoud-youssef)

- [Hochschule Bonn-Rhein-Sieg](https://www.h-brs.de/de)
