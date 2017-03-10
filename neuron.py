# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))


def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.  Всё как в лекции, никаких
    хитростей.

    neuron - нейрон, у которого есть метод vectorized_forward_pass,
    предсказывающий значения на выборке X X - матрица входных активаций (n, m) y
    - вектор правильных ответов (n, 1) Возвращает значение J (число) """

    assert y.shape[1] == 1, 'Incorrect y shape'

    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)


def J_quadratic_derivative(y, y_hat):
    """ Вычисляет вектор частных производных целевой функции по каждому из
    предсказаний.  y_hat - вертикальный вектор предсказаний, y - вертикальный
    вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим
    поэкспериментировать с целевыми функциями - полезно вынести эти вычисления в
    отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера
    отдельно.  """

    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)


def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """ Аналитическая производная целевой функции neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма
    квадратов отклонений y - правильные ответы для примеров из матрицы X J_prime
    - функция, считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1) """

    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах

    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat)
    dyhat_dz = neuron.activation_function_derivative(z)

    # осознайте эту строчку:
    dz_dw = X

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

    # можно было написать в два этапа. Осознайте, почему получается одно и то же
    # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
    # grad = np.sum(, axis=0)

    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T

    return grad


class Neuron():
    def __init__(self,
                 weights,
                 activation_function=sigmoid,
                 activation_function_derivative=sigmoid_prime):
        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def summator_single(self, single_input):
        return float(self.w.T.dot(single_input))

    def forward_pass(self, single_input):
        return self.activation_function(self.summator_single(single_input))

    def summatory(self, input_matrix):
        return input_matrix.dot(self.w)

    def activation(self, summatory_activation):
        return self.activation_function(summatory_activation)

    def vectorized_forward_pass(self, input_matrix):
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        pass

    def update_mini_batch(self, X, y, learning_rate, eps):
        J0 = J_quadratic(self, X, y)
        grad = compute_grad_analytically(self, X, y)
        self.w -= grad
        J1 = J_quadratic(self, X, y)
        if abs(J0-J1) < eps:
            return 1
        else:
            return 0


if __name__ == "__main__":
    np.random.seed(42)
    n = 10
    m = 5

    X = 20*np.random.sample((n, m))-10
    y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
    w = 2 * np.random.random((m, 1)) - 1
    neuron = Neuron(w)
    neuron.update_mini_batch(X, y, 0.1, 1e-5)
    print(neuron.w)
