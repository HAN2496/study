import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# 시뮬레이션 파라미터
dt = 0.01  # 시간 간격
num_steps = 1000  # 총 스텝 수

# 초기 조건
x1 = tf.Variable(10)
x2 = tf.Variable(10)
x1_t = tf.Variable(0)
x2_t = tf.Variable(0)

# 시간에 따른 각도와 각속도를 저장할 리스트
x1_list = []
x2_list = []
x1_t_list = []
x2_t_list = []
kinetic_energy_list = []
potential_energy_list = []
total_energy_list = []

# 시뮬레이션 루프
for step in range(num_steps):
    with tf.GradientTape(persistent=True) as tape:
        # 라그랑지안 계산
        T = kinetic_energy2(x1_t, x2_t)
        V = potential_energy2(x1, x2)
        L = lagrangian(T, V)

        dL_dq_dot1 = tape.gradient(L, x1_t)
    dL_dq_dot2 = tape.gradient(L, x2_t)
    del tape
    dL_dq_dot = dL_dq_dot1 + dL_dq_dot2

    with tf.GradientTape() as tape:
        T = kinetic_energy2(x1_t, x2_t)
        V = potential_energy2(x1, x2)
        L = lagrangian(T, V)

        dL_dq1 = tape.gradient(L, x1)
    dL_dq2 = tape.gradient(L, x2)
    x1_tt = tape.gradient()
    del tape
    dL_dq = dL_dq1 + dL_dq2
    # 오일러-라그랑주 방정식을 이용한 각속도의 업데이트

    theta_ddot = - dL_dtheta / (m * l**2)

    # 각속도와 각도 업데이트
    x1_t.assign_add(theta_ddot * dt)
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