# Evolutionary Robotics – Task Sheet 1  
**Winter Semester 2025**  
**Prof. Dr. Javad Ghofrani**  
**Deadline:** 22.10.2025  

---

## Objectives
- Implement a simple robot simulator and test it with Braitenberg vehicles.  
- Implement proximity sensors and control a robot using simple rules.  
- Explore fear and aggression behaviors (Task 1) and collision avoidance/exploration in bounded environments (Task 2).  

---

## Task 1: Simulator and Braitenberg Vehicles

### Overview
- Simulate **Braitenberg Vehicle 2** (fear and aggression).  
- Use a **torus-shaped space** (Pac-Man-like) where exiting one side re-enters the opposite.  
- Robots are represented by:
  - Position `(x, y)`  
  - Heading (direction)  
  - Speed  

### Light Source
- Fixed point on the torus.  
- Generates a **cone-shaped light intensity field** decreasing linearly with distance.  
- Robots detect light intensity at sensor positions.

### Sensors
- Two front sensors: left and right, relative to robot heading.  
- Sensor readings (`sl`, `sr`) used to determine motor control.

### Motor Control (Differential Drive)
- Velocities for left (`vl`) and right (`vr`) wheels are proportional to sensor readings:
  - **Aggressive vehicle:** `vl = sr`, `vr = sl` → approaches light.  
  - **Fearful vehicle:** `vl = sl`, `vr = sr` → avoids light.  
- Heading change: `Δheading = c * (vr - vl)`.  

### Output
- Live simulator shows robot motion and light field.  
- After simulation, **plots of trajectories** is generated.  
- Trajectory plot saved as: `braitenberg_vehicle_trajectories.png`.

---

## Task 2: Proximity Sensors and Rule-Based Control

### Environment
- 800 × 600 px rectangular arena with fixed outer walls.
- Includes multiple **internal and edge-connected line obstacles**.
- The robot starts **at the center of the arena**.

### Proximity Sensors
- Three **proximity sensors**: front-left, front, and front-right.
- Each sensor emits a ray (range = 15 % of arena width).
- Sensor value = `1 - (distance / range)` if wall detected, else 0..

### Controller
- Simple **if-then rules** (no learning/evolution yet) for:
  - If the front sensor is strong → slow down and turn toward the weaker side.
  - If one side sensor is active → turn away from that side.
  - Otherwise → move forward with small random heading variation.
- Robot heading updated based on sensor conditions.
- Collision avoidance  
- Exploration of the environment  

### Collision Handling
- The robot uses a **swept-circle (capsule) collision test** to avoid “skimming” through walls.
- A **tangent-projection method** allows smooth sliding along walls.

### Testing
- Initialize robots with different positions/headings.  
- Ensure no collisions or leaving the arena.

### Output
- Live simulation visualizes robot, sensors, and walls.  
- After simulation, **plots of trajectories** is generated.  
- Trajectory plot saved as: `trajectory_plot.png`.

---

## Usage Instructions

### Running Task 1 Simulator
```bash
python braitenberg_sim.py
```
- Runs 3-minute simulation (configurable).

- Visual preview shows robot movement and light intensity.

- After completion, trajectory and sensor intensity plots are generated automatically.

### Running Task 2 Simulator
```bash
python Task_2_v2.py
```
## 👥 Contributors:
- [Trushar Ghanekar](https://github.com/Trushar2411)
- [Ayushi Arora](https://github.com/ayushii206)
## Acknowledgements:
- [Prof. Dr. Javad Ghofrani](https://www.h-brs.de/de/inf/prof-dr-javad-ghofrani)

- [Youssef Mahmoud Youssef](https://www.h-brs.de/de/inf/youssef-mahmoud-youssef)

- [Hochschule Bonn-Rhein-Sieg](https://www.h-brs.de/de)