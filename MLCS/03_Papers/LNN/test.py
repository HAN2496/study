import pickle
import tensorflow as tf
file_path = "./save_weights/agent.pkl"

with open(file_path, 'rb') as f:
    agent = pickle.load(f)

inital_condition = [0, 0, 0]

# 예측을 위한 입력 데이터 예시 (두 개의 예측 샘플)
input_data = tf.convert_to_tensor([[0.1, 0.2], [0.5, 0.6]], dtype=tf.float32)
predicted_q_ddot = agent.physics_net(input_data)