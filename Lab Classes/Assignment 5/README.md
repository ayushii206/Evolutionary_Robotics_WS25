# Evolutionary Robotics – Task Sheet 4  
**Winter Semester 2025**  
**Prof. Dr. Javad Ghofrani**  
**Deadline:** 17.12.2025  

---

# Overview

This assignment focuses on behavioral attractors and internal dynamics in evolutionary robotics.

Goal: To evolve a neural network controller that enables a robot to survive autonomously by navigating toward a charging area and managing its internal energy level.

This assignment consists of the following task: Survival by Autonomous Recharging 

- Navigating robot in empty arena
- Survival behavior which is battery driven 
- EA evolved recurrent neural network controller  
- Analysis of emergent behavioral attractors

---

# Task — Survival by Autonomous Recharging

## Objective  
Evolve an Artificial Neural Network (ANN) controller that allows a mobile robot to:  

- Explore empty arena
- Monitor and manage battery level
- Navigate to charge station when low battery
- Recharge Autonomously and resume exploration

---

## Simulation Environment 

1. Continuous 2D arena (no walls)
2. Arena size: configurable (default: 10 × 10 units)
3. One corner contains:
   - A lamp emitting a light intensity field
   - A black floor area representing the charging station
4. Robot battery:
   - Normalized range: b ∈ [0, bmax]
   - Linear discharge over time
   - Instant recharge when entering charging area

---

## Robot Sensors

The robot is equipped with the following sensors:

- 3 proximity sensors: Front, Left (90°), Right (90°)
- 2 ambient light sensors: Front and Back
- 1 ground sensor: 0 (black charging area), 1 (normal floor)
- 1 battery sensor: Normalized battery level
- Total ANN inputs: 7

---

## Results

1. Fitness Evolution: `fitness.png`  
   
   - best fitness per generation
   - average fitness per generation
   - the curve shows improvement over generations followed by convergence  

2. Robot Trajectory: `trajectory.png`  
   
   - 2D plot of robot movement
   - charging area clearly marked

3. Battery level over time: `battery.png`  
   
   - gradual discharge during exploration
   - confirms successful utonomous recharging behavior



---

# Conclusion

- Evolution successfully produced autonomous survival behavior
- Recurrent neural networks enabled internal dynamics and memory
- Behavioral attractors emerged naturally through evolution
- The robot reliably balances exploration and recharging
- This task highlights the challenges of evolving adaptive, temporally extended behaviors in robotics.

## 👥 Contributors:

- [Ayushi Arora](https://github.com/ayushii206)

## Acknowledgements:

- [Prof. Dr. Javad Ghofrani](https://www.h-brs.de/de/inf/prof-dr-javad-ghofrani)

- [Youssef Mahmoud Youssef](https://www.h-brs.de/de/inf/youssef-mahmoud-youssef)

- [Hochschule Bonn-Rhein-Sieg](https://www.h-brs.de/de)
