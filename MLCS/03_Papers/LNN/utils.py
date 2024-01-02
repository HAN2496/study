import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os


m=1
l=1
g=-9.81

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
def kinetic_energy(q_dot, m=m, l=l):
    return 1/2 * m * (l * q_dot) ** 2

def potential_energy(theta, m=m, l=l):
    return m * g * l * (1 - tf.cos(theta))

def kinetic_energy2(x1_t, x2_t, m1=m, m2=m):
    return 1/2 * m1 * x1_t ** 2 + 1/2 * m2 * (x2_t + x1_t) ** 2

def potential_energy2(x1, x2, m1=m, m2=m):
    return m1 * g * x1 + m2 * g * (x1 + x2) + 1/2 * k * x1^2

def damper_energy(x1_t, x2_t):
    return - c * (x2_t - x1_t)

def lagrangian(T, V):
    return T - V

#def la

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

class LagrangianNN(Model):
    def __init__(self, layer_sizes):
        super(LagrangianNN, self).__init__()
        self.layers_list = [Dense(size, activation='relu') for size in layer_sizes]
        self.output_layer = Dense(1, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)

class LNN(object):
    def __init__(self, layer_sizes):
        self.model = LagrangianNN(layer_sizes)
        self.optimizer = Adam()

    def train_step(self, q, q_dot, q_ddot_true):
        with tf.GradientTape() as tape:
            L = self.model(tf.concat([q, q_dot], axis=1))
            L_q = tf.gradients(L, q)[0]
            L_q_dot = tf.gradients(L, q_dot)[0]

            q_ddot_pred = tf.gradients(L_q_dot, q_dot)[0] - L_q
            loss = tf.reduce_mean(tf.square(q_ddot_true - q_ddot_pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, q, q_dot, q_ddot, epochs):
        for epoch in range(epochs):
            loss = self.train_step(q, q_dot, q_ddot)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    def predict(self, q, q_dot):
        L = self.model(tf.concat([q, q_dot], axis=1))
        L_q = tf.gradients(L, q)[0]
        L_q_dot = tf.gradients(L, q_dot)[0]
        q_ddot_pred = tf.gradients(L_q_dot, q_dot)[0] - L_q
        return q_ddot_pred
