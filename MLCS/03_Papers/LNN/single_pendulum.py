import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# 시뮬레이션 파라미터
dt = 0.01  # 시간 간격
num_steps = 1000  # 총 스텝 수

# 초기 조건
theta = tf.Variable(-np.pi/2)  # 초기 각도 (45도)
theta_dot = tf.Variable(0.0)    # 초기 각속도

q=theta
qdot = theta_dot

# 시간에 따른 각도와 각속도를 저장할 리스트
theta_list = []
theta_dot_list = []
kinetic_energy_list = []
potential_energy_list = []
total_energy_list = []

# 시뮬레이션 루프
for step in range(num_steps):
    with tf.GradientTape() as tape:
        # 라그랑지안 계산
        T = kinetic_energy(theta_dot)
        V = potential_energy(theta)
        L = lagrangian(T, V)

    # 각속도에 대한 라그랑지안의 미분
    dL_dtheta_dot = tape.gradient(L, theta_dot)

    with tf.GradientTape() as tape:
        T = kinetic_energy(theta_dot)
        V = potential_energy(theta)
        L = lagrangian(T, V)

    # 각도에 대한 라그랑지안의 미분
    dL_dtheta = tape.gradient(L, theta)
    # 오일러-라그랑주 방정식을 이용한 각속도의 업데이트
    theta_ddot = - dL_dtheta / (m * l**2)

    # 각속도와 각도 업데이트
    theta_dot.assign_add(theta_ddot * dt)
    theta.assign_add(theta_dot * dt)

    # 결과 저장
    theta_list.append(theta.numpy())
    theta_dot_list.append(theta_dot.numpy())
    kinetic_energy_list.append(T)
    potential_energy_list.append(V)
    total_energy_list.append(T+V)

# 결과 플롯
plt.subplot(2,2,1)
plt.plot(theta_list, label='Theta')
plt.legend()
plt.subplot(2,2,2)
plt.plot(theta_dot_list, label='Theta dot')
plt.legend()
plt.subplot(2,2,3)
plt.plot(kinetic_energy_list, label='T')
plt.plot(potential_energy_list, label='V')
plt.legend()
plt.subplot(2,2,4)
plt.plot(total_energy_list, label='Total')
plt.legend()
plt.show()


"""
Pygame으로 시각화 하는 부분
"""
import pygame
import math

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Pendulum Simulation")

WHITE = (255, 255, 255)
RED = (255, 0, 0)

l = 200  # 막대 길이. 시각화라 그냥 크게 그려도 괜찮음.
r = 10   # 막대 끝 구의 반지름
origin = (width // 2, height // 4)  # 고정점의 위치

# 시뮬레이션 루프
running = True
clock = pygame.time.Clock()

for theta in theta_list:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # 단진자 위치 계산
    x = origin[0] + l * math.sin(theta)
    y = origin[1] + l * math.cos(theta)

    pygame.draw.line(screen, RED, origin, (x, y), 2)
    pygame.draw.circle(screen, RED, (int(x), int(y)), r)

    pygame.display.flip()
    clock.tick(60)
    if not running:
        break
pygame.quit()