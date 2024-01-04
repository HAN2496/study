import tensorflow as tf
import numpy as np

# Define the variables
x = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)
y = tf.Variable([4.0, 5.0, 6.0], dtype=tf.float32)

# Define the function L(x, y) for a single element
def L_single(x, y):
    return x**2 * y + y**3

# Vectorized function L(x, y) for the whole vector
def L_vectorized(x, y):
    return tf.map_fn(lambda args: L_single(*args), (x, y), dtype=tf.float32)

# Compute the second order mixed partial derivative for each element
def compute_second_order_derivative(x, y):
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(y)
            L_values = L_vectorized(x, y)

        dL_dy = tape2.gradient(L_values, y)

    return tape1.gradient(dL_dy, x)

# Calculate the second order mixed partial derivatives
second_order_derivatives = compute_second_order_derivative(x, y)

# Print the results
print(second_order_derivatives.numpy())

def single_pendulum_single(q, q_t):
    m = 1.0
    l = 1.0
    g = 10.0
    return 1 / 2 * m * (l * q_t) ** 2 - m * g * l * (1 - tf.cos(q))

def single_pendulum(q, q_t):
    return tf.map_fn(lambda args: single_pendulum_single(*args), (q, q_t), dtype=tf.float32)

tf_state = tf.Variable([[0.0, np.pi/6, np.pi/3, np.pi/2],
                       [1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)

# Note: To execute this function, you need the 'lagrangian' function and a valid 'state' vector.
# This code assumes TensorFlow is installed and the necessary variables and functions are defined.

def tensorflow_lagrangian_eom_modified_v2(lagrangian, state):
    q, q_t = state[0], state[1]

    # Use persistent GradientTape for multiple gradient calculations
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([q, q_t])
        L = lagrangian(q, q_t)
        L_q = tape.gradient(L, q)
        L_q_t = tape.gradient(L, q_t)

    # Compute the Jacobian of L_q_t with respect to q_t (Hessian)
    L_q_t_q_t = tape.jacobian(L_q_t, q_t)

    # Compute the Jacobian of L_q with respect to q_t
    L_q_q_t = tape.jacobian(L_q, q_t)

    # Handle the case where L_q_q_t is None
    if L_q_q_t is None:
        L_q_q_t = tf.zeros_like(L_q_t_q_t)

    # Calculate q_ddot
    q_ddot = tf.linalg.inv(L_q_t_q_t) @ (tf.reshape(L_q, [-1, 1]) - tf.matmul(L_q_q_t, tf.reshape(q_t, [-1, 1])))

    # Delete the persistent tape
    del tape

    return q_ddot

# Test the modified function
result_modified = tensorflow_lagrangian_eom_modified_v2(single_pendulum, tf_state)

# This will compute without the previous runtime error
print(result_modified.numpy())



# 함수 호출
result = tensorflow_lagrangian_eom_modified_v2(single_pendulum, tf_state)