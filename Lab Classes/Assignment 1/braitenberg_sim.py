import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os

# --- Parameters ---
WIDTH, HEIGHT = 800, 600
FPS = 60
SIM_TIME = 180  # 3 minutes
VISUAL_TIME = 20  # seconds of visible simulation before running headless
ROBOT_RADIUS = 15
SENSOR_OFFSET = 30
SENSOR_SPREAD = math.radians(45)
LIGHT_INTENSITY_MAX = 255
LIGHT_DECAY = 0.7
LIGHT_POS = (WIDTH // 2, HEIGHT // 2)

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Braitenberg Vehicle 2 (Aggression & Fear) — Preview")
clock = pygame.time.Clock()

# --- Utility functions ---
def torus_distance(x1, y1, x2, y2):
    dx = min(abs(x1 - x2), WIDTH - abs(x1 - x2))
    dy = min(abs(y1 - y2), HEIGHT - abs(y1 - y2))
    return math.sqrt(dx**2 + dy**2)

def get_light_intensity(x, y, light_pos):
    dist = torus_distance(x, y, *light_pos)
    intensity = max(LIGHT_INTENSITY_MAX - LIGHT_DECAY * dist, 0)
    return intensity / 255  # normalized 0..1

def generate_light_surface():
    surface = pygame.Surface((WIDTH, HEIGHT))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            intensity = get_light_intensity(x, y, LIGHT_POS)
            color = (int(255 * intensity), int(180 * intensity), 0)
            surface.set_at((x, y), color)
    return surface

light_surface = generate_light_surface()

# --- Robot class ---
class Robot:
    def __init__(self, x, y, color, behavior="aggressive"):
        self.x = x
        self.y = y
        self.heading = random.uniform(0, 2 * math.pi)
        self.color = color
        self.behavior = behavior
        self.turn_gain = 0.2
        self.speed_gain = 3.0
        self.path = []
        self.light_history = []

    def sensor_positions(self):
        left_angle = self.heading + SENSOR_SPREAD
        right_angle = self.heading - SENSOR_SPREAD
        sl_x = (self.x + SENSOR_OFFSET * math.cos(left_angle)) % WIDTH
        sl_y = (self.y + SENSOR_OFFSET * math.sin(left_angle)) % HEIGHT
        sr_x = (self.x + SENSOR_OFFSET * math.cos(right_angle)) % WIDTH
        sr_y = (self.y + SENSOR_OFFSET * math.sin(right_angle)) % HEIGHT
        return (sl_x, sl_y), (sr_x, sr_y)

    def update(self, light_pos):
        (sl_x, sl_y), (sr_x, sr_y) = self.sensor_positions()
        sl = get_light_intensity(sl_x, sl_y, light_pos)
        sr = get_light_intensity(sr_x, sr_y, light_pos)

        # --- Braitenberg logic ---
        if self.behavior == "aggressive":
            vl = sr
            vr = sl
        else:  # fearful
            vl = 1 - sl
            vr = 1 - sr

        # Heading change
        self.heading += self.turn_gain * (vr - vl)
        self.heading %= 2 * math.pi

        # Forward motion
        v = self.speed_gain * (vl + vr) / 2
        self.x = (self.x + v * math.cos(self.heading)) % WIDTH
        self.y = (self.y + v * math.sin(self.heading)) % HEIGHT

        # Record data
        self.path.append((self.x, self.y))
        avg_light = get_light_intensity(self.x, self.y, light_pos)
        self.light_history.append(avg_light)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), ROBOT_RADIUS)
        hx = self.x + 20 * math.cos(self.heading)
        hy = self.y + 20 * math.sin(self.heading)
        pygame.draw.line(surface, (255, 255, 255), (self.x, self.y), (hx, hy), 2)

# --- Initialize robots ---
aggressor = Robot(200, 300, (255, 80, 80), "aggressive")
fearful = Robot(600, 300, (100, 100, 255), "fearful")

# --- Run visible simulation briefly ---
visible_frames = VISUAL_TIME * FPS
running = True
frame_count = 0
while running and frame_count < visible_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    aggressor.update(LIGHT_POS)
    fearful.update(LIGHT_POS)

    # Draw scene
    screen.blit(light_surface, (0, 0))
    pygame.draw.circle(screen, (255, 255, 0), (int(LIGHT_POS[0]), int(LIGHT_POS[1])), 10)
    aggressor.draw(screen)
    fearful.draw(screen)
    pygame.display.flip()

    clock.tick(FPS)
    frame_count += 1

pygame.quit()
print("Running remaining simulation headlessly for full 5-minute trajectory...")

# --- Continue simulation silently for full 5 min ---
total_frames = SIM_TIME * FPS
for frame in range(frame_count, total_frames):
    aggressor.update(LIGHT_POS)
    fearful.update(LIGHT_POS)

print("Simulation complete. Generating plots...")

# --- Compute light field for plotting ---
X, Y = np.meshgrid(np.linspace(0, WIDTH, 100), np.linspace(0, HEIGHT, 100))
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = get_light_intensity(X[i, j], Y[i, j], LIGHT_POS)

# --- Plot 1: Trajectories on light field ---
plt.figure(figsize=(8, 6))
plt.title("Braitenberg Vehicle 2 — Full 3-Minute Trajectories")
plt.imshow(Z, extent=(0, WIDTH, 0, HEIGHT), origin='lower', cmap='hot', alpha=0.8)
plt.scatter(LIGHT_POS[0], LIGHT_POS[1], color='yellow', s=100, edgecolors='black', label='Light Source')

ax, ay = zip(*aggressor.path)
fx, fy = zip(*fearful.path)
plt.plot(ax, ay, color='red', label='Aggressive Vehicle → approaches light')
plt.plot(fx, fy, color='blue', label='Fearful Vehicle → avoids light')

plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()

output_folder = "Lab Classes/Assignment 1/output"  
os.makedirs(output_folder, exist_ok=True)

filename = "braitenberg_vehicle_trajectories.png"
filepath = os.path.join(output_folder, filename)
plt.savefig(filepath)
  
plt.show()


