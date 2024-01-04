import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# System Parameters
m1 = 1.0  # Mass of M1
m2 = 1.0  # Mass of M2
k = 50.0  # Spring constant
b = 10.0  # Damping constant
g = -9.81  # Gravity

# Define the system dynamics
def system_dynamics(x1, x1_dot, x2, x2_dot):
    # Spring force (F = -kx)
    spring_force = -k * x1
    # Damping force (F = -bv)
    damping_force = -b * (x2_dot - x1_dot)
    # Gravitational force
    gravity_m1 = m1 * g
    gravity_m2 = m2 * g

    # Newton's second law for each mass
    x1_ddot = (spring_force - damping_force) / m1
    x2_ddot = (damping_force - gravity_m2) / m2

    return x1_ddot, x2_ddot

# Simulation parameters
dt = 0.01  # Time step
num_steps = 1000  # Number of simulation steps

# Initial conditions
x1 = tf.Variable(0.0)  # Initial displacement of M1
x1_dot = tf.Variable(0.0)  # Initial velocity of M1
x2 = tf.Variable(0.0)  # Initial displacement of M2
x2_dot = tf.Variable(0.0)  # Initial velocity of M2

# Lists to store simulation results
x1_list, x1_dot_list, x2_list, x2_dot_list = [], [], [], []

# Simulation loop
for step in range(num_steps):
    # Update dynamics
    x1_ddot, x2_ddot = system_dynamics(x1, x1_dot, x2, x2_dot)

    # Euler integration
    x1_dot.assign_add(x1_ddot * dt)
    x1.assign_add(x1_dot * dt)
    x2_dot.assign_add(x2_ddot * dt)
    x2.assign_add(x2_dot * dt)

    # Store results
    x1_list.append(x1.numpy())
    x1_dot_list.append(x1_dot.numpy())
    x2_list.append(x2.numpy())
    x2_dot_list.append(x2_dot.numpy())

# Plotting the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x1_list, label='x1 (Displacement of M1)')
plt.plot(x2_list, label='x2 (Displacement of M2)')
plt.title('Displacements over Time')
plt.xlabel('Time step')
plt.ylabel('Displacement')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x1_dot_list, label='x1_dot (Velocity of M1)')
plt.plot(x2_dot_list, label='x2_dot (Velocity of M2)')
plt.title('Velocities over Time')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()

import pygame
import math

# Pygame 초기화
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Two-Mass Spring-Damper System Visualization")

# 색상 설정
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 시각화 파라미터
origin = (width // 2, height // 2)  # 중심점 위치
mass_radius = 20  # 질량 표시 원의 반지름
spring_width = 2  # 스프링 선의 너비
damper_width = 2  # 댐퍼 선의 너비
scale_factor = 50  # 변위 스케일 조정 인자

# 시각화 루프
running = True
clock = pygame.time.Clock()
for i in range(num_steps):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # 질량 위치 계산
    y1 = origin[1] + x1_list[i] * scale_factor
    y2 = y1 + x2_list[i] * scale_factor

    # 스프링 그리기 (천장에서 M1까지)
    pygame.draw.line(screen, GREEN, origin, (origin[0], y1), spring_width)

    # M1 그리기
    pygame.draw.circle(screen, RED, (origin[0], int(y1)), mass_radius)

    # 댐퍼 그리기 (M1에서 M2까지)
    pygame.draw.line(screen, BLUE, (origin[0], y1), (origin[0], y2), damper_width)

    # M2 그리기
    pygame.draw.circle(screen, RED, (origin[0], int(y2)), mass_radius)

    pygame.display.flip()
    clock.tick(60)  # 프레임레이트 설정

    if not running:
        break

pygame.quit()

