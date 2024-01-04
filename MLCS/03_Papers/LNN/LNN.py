from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from utils import *

dt = 0.01
m=1
l=1
g=-9.81


class SinglePendulum:
    def __init__(self):
        self.m = 1
        self.l = l
        self.num_steps = 1000
        self.theta = tf.Variable(-np.pi/2)
        self.theta_dot = tf.Variable(0.0)
        self.theta_list = []
        self.theta_dot_list = []
        self.kinetic_energy_list = []
        self.potential_energy_list = []
        self.total_energy_list = []
        self.calc()

    def kinetic_energy(self, theta_dot):
        return 1/2 * m * (l * theta_dot) ** 2

    def potential_energy(self, theta):
        return m * g * l * (1 - tf.cos(theta))

    def lagrangian(self, T, V):
        return T - V

    def calc(self):
        for step in range(self.num_steps):
            with tf.GradientTape() as tape:
                # 라그랑지안 계산
                T = self.kinetic_energy(self.theta_dot)
                V = self.potential_energy(self.theta)
                L = self.lagrangian(T, V)

            # 각속도에 대한 라그랑지안의 미분
            dL_dtheta_dot = tape.gradient(L, self.theta_dot)

            with tf.GradientTape() as tape:
                T = self.kinetic_energy(self.theta_dot)
                V = self.potential_energy(self.theta)
                L = self.lagrangian(T, V)

            # 각도에 대한 라그랑지안의 미분
            dL_dtheta = tape.gradient(L, self.theta)
            # 오일러-라그랑주 방정식을 이용한 각속도의 업데이트
            theta_ddot = - dL_dtheta / (m * l ** 2)

            # 각속도와 각도 업데이트
            self.theta_dot.assign_add(theta_ddot * dt)
            self.theta.assign_add(self.theta_dot * dt)

            # 결과 저장
            self.theta_list.append(self.theta.numpy())
            self.theta_dot_list.append(self.theta_dot.numpy())
            self.kinetic_energy_list.append(T)
            self.potential_energy_list.append(V)
            self.total_energy_list.append(T + V)

    def calc_q_ddot(self, theta):
        return - g / l * tf.sin(theta)
    def show(self):
        plt.subplot(2, 2, 1)
        plt.plot(self.theta_list, label='Theta')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(self.theta_dot_list, label='Theta dot')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(self.kinetic_energy_list, label='T')
        plt.plot(self.potential_energy_list, label='V')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(self.total_energy_list, label='Total')
        plt.legend()
        plt.show()

class Lagrangian(Model):
    def __init__(self, layer_sizes=6, node_num=20):
        super(Lagrangian, self).__init__()
        self.layers_list = [Dense(node_num, activation='relu') for _ in range(layer_sizes)]
        self.output_layer = Dense(1, activation='linear')

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)

class LNN(object):
    def __init__(self, lagrangian, layer_sizes):
        self.lagrangian = lagrangian
        self.layer_sizes = layer_sizes
        self.lr = 0.001
        self.optimizer = Adam(self.lr)

        self.model = Lagrangian(layer_sizes)
        self.model.build(input_shape=(None, 2))

    def physics_net(self, state):
        q, q_t = state[:, 0:1], state[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([q, q_t])
            L = self.lagrangian(q, q_t)
            L_q = tape.gradient(L, q)
            L_q_t = tape.gradient(L, q_t)
        L_q_t_q_t = tape.jacobian(L_q_t, q_t)
        L_q_q_t = tape.jacobian(L_q, q_t)
        tape.jacobian(L_q, q_t) if L_q_q_t is not None else tf.zeros_like(L_q_t_q_t)

        q_ddot = tf.linalg.inv(L_q_t_q_t) @ (tf.reshape(L_q, [-1, 1]) - tf.matmul(L_q_q_t, tf.reshape(q_t, [-1, 1])))
        del tape

        return q_ddot

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(path + 'burgers.h5')

        with open(path + 'agent.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load_weights(self, path):
        self.model.load_weights(path + 'burgers.h5')

    def learn(self, state, q_ddot_true):
        with tf.GradientTape() as tape:
            f = self.physics_net(state)
            loss = tf.reduce_mean(tf.square(f - q_ddot_true))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def train(self, max_num):
        q_data = np.random.uniform(-np.pi/2, np.pi/2, [20000, 1])
        q_dot_data = np.random.uniform(-np.pi/2, np.pi/2, [20000, 1])
        state = np.concatenate((q_data, q_dot_data), axis=1)
        q_ddot_true_data = np.zeros_like(q_data)
        for idx, q in enumerate(q_data):
            q_ddot_true_data[idx] = single_pendulum_qddot(q)

        train_loss_history = []
        print("*"*50)
        print(f"hidden layer num: {self.layer_sizes}")
        for iter in range(int(max_num)):
            loss = self.learn(tf.convert_to_tensor(state, dtype=tf.float32),
                              tf.convert_to_tensor(q_ddot_true_data, dtype=tf.float32))

            train_loss_history.append([iter, loss.numpy()])

            if iter % 300 == 0:
                print('iter=', iter, ', loss=', loss.numpy())
        self.save_weights("./save_weights/")

        np.savetxt(f'./save_weights/loss{self.layer_sizes}.txt', train_loss_history)
        train_loss_history = np.array(train_loss_history)
        self.train_loss_history = train_loss_history

def main():

    max_num = 3000
    lagrangian = single_pendulum
    agent = LNN(lagrangian, layer_sizes=6)
    agent.train(max_num)
    min_loss = np.min(agent.train_loss_history)
    agent.save_weights("./save_weights/")

if __name__=="__main__":
    main()