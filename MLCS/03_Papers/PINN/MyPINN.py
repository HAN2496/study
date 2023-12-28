# PINN for Burgers' equation
# coded by St.Watermelon

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

class Burgers(Model):

    def __init__(self, h_num=6, node_num=20):
        super(Burgers, self).__init__()
        if h_num <= 0: raise ValueError("Put positive integer number")

        self.h_num = h_num
        self.h_arr = [Dense(node_num, activation='tanh') for _ in range(hs)]
        self.u = Dense(1, activation='linear')


    def call(self, state):
        for idx, h in enumerate(h_arr):
            if idx == 0:
                x = self.h_arr[i](state)
            else:
                x = self.h_arr[i](x)
        out = self.u(x)
        return out


class Pinn(object):

    def __init__(self, h_num=6, node_num=20):
        self.h_num = h_num
        self.node_num = node_num

        self.lr = 0.001
        self.opt = Adam(self.lr)

        self.burgers = Burgers(h_num=h_num, node_num=node_num)
        self.burgers.build(input_shape=(None, 2))

    def physics_net(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)
            xt_t = tf.concat([x, t], axis=1)
            u = self.burgers(xt_t)
            u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)
        u_t = tape.gradient(u, t)
        del tape

        return u_t + u*u_x - (0.01/np.pi)*u_xx


    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.burgers.save_weights(path + 'burgers.h5')


    def load_weights(self, path):
        self.burgers.load_weights(path + 'burgers.h5')


    def learn(self, xt_col, xt_bnd, tu_bnd):
        with tf.GradientTape() as tape:
            f = self.physics_net(xt_col)
            loss_col = tf.reduce_mean(tf.square(f))

            tu_bnd_hat = self.burgers(xt_bnd)
            loss_bnd = tf.reduce_mean(tf.square(tu_bnd_hat-tu_bnd))

            loss = loss_col + loss_bnd

        grads = tape.gradient(loss, self.burgers.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.burgers.trainable_variables))

        return loss


    def predict(self, xt):
        tu = self.burgers(xt)
        return tu


    def train(self, max_num):

        # initial and boundary condition
        x_data = np.linspace(-1.0, 1.0, 500)
        t_data = np.linspace(0.0, 1.0, 500)
        xt_bnd_data = []
        tu_bnd_data = []

        for x in x_data:
            xt_bnd_data.append([x, 0]) # IC
            tu_bnd_data.append([-np.sin(np.pi * x)]) # BC

        for t in t_data:
            xt_bnd_data.append([1, t]) # BC
            tu_bnd_data.append([0])
            xt_bnd_data.append([-1, t]) # BC
            tu_bnd_data.append([0])

        xt_bnd_data = np.array(xt_bnd_data)
        tu_bnd_data = np.array(tu_bnd_data)

        # collocation point
        t_col_data = np.random.uniform(0, 1, [20000, 1])
        x_col_data = np.random.uniform(-1, 1, [20000, 1])
        xt_col_data = np.concatenate([x_col_data, t_col_data], axis=1)
        xt_col_data = np.concatenate((xt_col_data, xt_bnd_data), axis=0)

        train_loss_history = []

        print("*"*50)
        print(f"hidden layer num: {self.hs}")
        time_before = datetime.now()
        for iter in range(int(max_num)):

            loss = self.learn(tf.convert_to_tensor(xt_col_data, dtype=tf.float32),
                       tf.convert_to_tensor(xt_bnd_data, dtype=tf.float32),
                       tf.convert_to_tensor(tu_bnd_data, dtype=tf.float32))

            train_loss_history.append([iter, loss.numpy()])

            if iter % 300 == 0:
                print('iter=', iter, ', loss=', loss.numpy())
        time_after = datetime.now()
        self.save_weights("./save_weights/")

        np.savetxt(f'./save_weights/loss{self.hs}.txt', train_loss_history)
        train_loss_history = np.array(train_loss_history)
        self.train_loss_history = train_loss_history

    def show(self):
        plt.plot(train_loss_history[:, 0], train_loss_history[:, 1])
        plt.yscale("log")
        plt.show()


def main():

    max_num = 3000
    loss_arr = []
    for i in range(10):
        i = i + 6
        agent = Pinn(h_num=i)
        agent.train(max_num)
        min_loss = np.min(agent.train_loss_history)
        loss_arr.append([i, min_loss])
    print(loss_arr)


if __name__=="__main__":
    main()