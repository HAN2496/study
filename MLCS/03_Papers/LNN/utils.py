import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os


m=1
l=1
g=9.81

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
def kinetic_energy(q_dot, m=m, l=l):
    return 1/2 * m * (l * q_dot) ** 2

def potential_energy(theta, m=m, l=l):
    return m * g * l * (1 - tf.cos(theta))

def lagrangian(T, V):
    return T - V

def acceleration(L, q, qdot):
    pass

class SimplePendulum(Model):

    def __init__(self, h_num=6, node_num=20):
        super(SimplePendulum, self).__init__()
        if h_num <= 0: raise ValueError("Put positive integer number")

        self.h_num = h_num
        self.h_arr = [Dense(node_num, activation='tanh') for _ in range(h_num)]
        self.u = Dense(1, activation='linear')


    def call(self, state):
        for idx, h in enumerate(self.h_arr):
            if idx == 0:
                x = self.h_arr[idx](state)
            else:
                x = self.h_arr[idx](x)
        out = self.u(x)
        return out

class LNN(object):
    def __init__(self):
        self.lr = 0.001
        self.opt = Adam(self.lr)

        self.simple_pendulum = SimplePendulum()
        self.simple_pendulum.build(input_shape=(None, 2))

    def physics_net(self):
        pass