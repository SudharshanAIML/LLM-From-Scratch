def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


class GradientDescent:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, weight, gradient):
        return weight - self.lr * gradient
def square(x):
    return x * x

print(numerical_gradient(square, 3))
# ≈ 6

gd = GradientDescent(0.1)

w = 3
grad = -8

print(gd.step(w, grad))
# 3.8