import tensorflow as tf

# 함수 정의, 예를 들어 f(x, y) = x^2 + xy + y^2
def f(x, y):
    return x**2 + x*y + y**2

# 변수 정의
x = tf.Variable(1.0)
y = tf.Variable(2.0)

# 헤시안 행렬 계산
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        z = f(x, y)
    dz_dx, dz_dy = tape1.gradient(z, [x, y])
hessian = [tape2.gradient(dz_dx, [x, y]), tape2.gradient(dz_dy, [x, y])]

# 결과 출력
for row in hessian:
    print([t.numpy() for t in row])
