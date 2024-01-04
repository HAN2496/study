import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

dt = 0.01
m=1
l=1
g=-9.81

m1=m
m2=m
k=100
c=10

"""

def lagrangian(L, q_dot):
    # L은 라그랑지안 함수, q_dot은 각속도 벡터

    # 각속도에 대한 라그랑지안의 그래디언트 계산
    grad_L_q_dot = jax.grad(L, argnums=1)(q, q_dot)

    # 그래디언트의 전치(transpose)
    grad_L_q_dot_T = jnp.transpose(grad_L_q_dot)

    # 전치된 그래디언트와 그래디언트의 행렬 곱셈(matrix mxultiplication)
    mass_matrix = jnp.matmul(grad_L_q_dot_T, grad_L_q_dot)

    # 역행렬 계산
    mass_matrix_inv = jnp.linalg.inv(mass_matrix)

    return mass_matrix_inv

lagrangian(1, [1,1])
"""

def single_pendulum_single(q, q_t):
    return 1 / 2 * m * (l * q_t) ** 2 - m * g * l * (1 - tf.cos(q))
def single_pendulum(q, q_t):
    return tf.map_fn(lambda args: single_pendulum_single(*args), (q, q_t), dtype=tf.float32)

def single_pendulum_qddot(q):
    return tf.cos(q) - 10 * tf.sin(q)