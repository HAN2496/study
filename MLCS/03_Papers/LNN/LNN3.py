import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
import os

m = 1.0
l = 1.0
g = 10.0

def single_pendulum_single(q, q_t):
    return 1 / 2 * m * (l * q_t) ** 2 - m * g * l * (1 - tf.cos(q))


def single_pendulum(q, q_t):
    return tf.map_fn(lambda args: single_pendulum_single(*args), (q, q_t), dtype=tf.float32)

def ex2_single(q, q_t):
    a = 1.0
    t = 1.0
    T = 1/2 * m * ((a * t - l * q_t * tf.cos(q)) ** 2 + (l * q_t * tf.sin(q)) ** 2)
    V = - m * g * l * tf.cos(q)
    return T-V

def ex2(q, q_t):
    return tf.map_fn(lambda args: ex2_single(*args), (q, q_t), dtype=tf.float32)

def ex1_single(q, q_t):
    m = 1
    M = 1
    xdot = 1
    ydot = 1
    y = 1
    T = 1/2 * m * (2 * xdot ** 2 + 2 * xdot * ydot * tf.cos(q) + ydot **2)
    V = - m * g * y * tf.sin(q)
    return T-V

def ex1(q, q_t):
    return tf.map_fn(lambda args: ex2_single(*args), (q, q_t), dtype=tf.float32)

def tensorflow_lagrangian_eom(lagrangian, state):
    q, q_t = state[0], state[1]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(q_t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(q_t)
            L = lagrangian(q, q_t)
        L_q_t = tape1.gradient(L, q_t)
    L_q_t_q_t = tape2.jacobian(L_q_t, q_t)

    with tf.GradientTape() as tape4:
        tape4.watch(q_t)
        with tf.GradientTape() as tape3:
            tape3.watch(q)
            L = lagrangian(q, q_t)
        L_q = tape3.gradient(L, q)
    L_q_q_t = tape4.jacobian(L_q, q_t)
    if L_q_q_t is None:
        L_q_q_t = tf.zeros_like(L_q_t_q_t)
    q_ddot = tf.linalg.inv(L_q_t_q_t) @ (tf.reshape(L_q, [-1, 1]) - tf.matmul(L_q_q_t, tf.reshape(q_t, [-1, 1])))
    return q_ddot

def lagrangian_eom(lagrangian, state):
    q, q_t = state[0], state[1]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([q, q_t])
        L = lagrangian(q, q_t)
        L_q = tape.gradient(L, q)
        L_q_t = tape.gradient(L, q_t)
    L_q_t_q_t = tape.jacobian(L_q_t, q_t)
    L_q_q_t = tape.jacobian(L_q, q_t)
    tape.jacobian(L_q, q_t) if L_q_q_t is not None else tf.zeros_like(L_q_t_q_t)

    q_ddot = tf.linalg.inv(L_q_t_q_t) @ (tf.reshape(L_q, [-1, 1]) - tf.matmul(L_q_q_t, tf.reshape(q_t, [-1, 1])))
    del tape

    return q_ddot

# 반드시 행이 2개여야 함
tf_state = tf.Variable([
                       [0.0, np.pi/6, np.pi/3, np.pi/2],
                       [0.0, 0.0, 0.0, 0.0]],
                       dtype=tf.float32)

result = lagrangian_eom(ex2, tf_state)
#result = tensorflow_lagrangian_eom(single_pendulum_single, tf_state)
print(result)