import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# 시뮬레이션 파라미터
dt = 0.01  # 시간 간격
num_steps = 1000  # 총 스텝 수

# 초기 조건
theta = tf.Variable(np.pi / 4)  # 초기 각도 (45도)
theta_dot = tf.Variable(0.0)    # 초기 각속도

q=theta
qdot = theta_dot

# 시간에 따른 각도와 각속도를 저장할 리스트
theta_list = []
theta_dot_list = []

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
        tape.watch(theta)
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

# 결과 플롯
plt.subplot(2,1,1)
plt.plot(theta_list, label='Theta')
plt.legend()
plt.subplot(2,1,2)
plt.plot(theta_dot_list, label='Theta dot')
plt.legend()
plt.show()
