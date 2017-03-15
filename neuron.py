# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3


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


def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
    # here goes your code
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):

        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] -= eps
        minus_cost = J(neuron, X, y)
        neuron.w[i] = old_wi

        neuron.w[i] += eps
        plus_cost = J(neuron, X, y)
        neuron.w[i] = old_wi
        # Считаем новое значение целевой функции и вычисляем приближенное
        # значение градиента
        num_grad[i] = (plus_cost - minus_cost)/(2*eps)

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать
        # ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad


def compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции neuron - объект класса Neuron X -
    вертикальная матрица входов формы (n, m), на которой считается сумма
    квадратов отклонений y - правильные ответы для тестовой выборки X J -
    целевая функция, градиент которой мы хотим получить eps - размер $\delta w$
    (малого изменения весов)
    """

    initial_cost = J(neuron, X, y)
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):

        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] += eps

        # Считаем новое значение целевой функции и вычисляем приближенное
        # значение градиента
        num_grad[i] = (J(neuron, X, y) - initial_cost)/eps

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать
        # ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad


def J_by_weights(weights, X, y, bias):
    """
    Посчитать значение целевой функции для нейрона с заданными весами.  Только
    для визуализации
    """
    new_w = np.hstack((bias, weights)).reshape((3, 1))
    return J_quadratic(Neuron(new_w), X, y)


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

    def forward_pass(self, summator):
        if summator > 0:
            return 1
        else:
            return 0

    def summatory(self, input_matrix):
        return input_matrix.dot(self.w)

    def activation(self, summatory_activation):
        return self.activation_function(summatory_activation)

    def vectorized_forward_pass(self, input_matrix):
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        data = np.concatenate((X, y), axis=1)
        step = 0
        while step < max_steps:
            np.random.shuffle(data)
            prev_ind = 0
            for ind in range(batch_size, data.shape[0]+1, batch_size):
                if self.update_mini_batch(data[prev_ind:ind, :-1],
                                          data[prev_ind:ind, -1:],
                                          0.1, eps) == 1:
                    return 1
                else:
                    prev_ind = ind
                    step += 1
        return 0

    def update_mini_batch(self, X, y, learning_rate, eps):
        J0 = J_quadratic(self, X, y)
        grad = compute_grad_analytically(self, X, y)
        self.w -= (grad * learning_rate)
        J1 = J_quadratic(self, X, y)
        if abs(J0-J1) < eps:
            return 1
        else:
            return 0


if __name__ == "__main__":
    print(p3.__name__)
    np.random.seed(42)
    n = 10
    m = 5

    X = 20*np.random.sample((n, m))-10
    y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
    w = 2 * np.random.random((m, 1)) - 1
    neuron = Neuron(w)
    neuron.update_mini_batch(X, y, 0.1, 1e-5)
    print(neuron.w)

    max_b = 40
    min_b = -40
    max_w1 = 40
    min_w1 = -40
    max_w2 = 40
    min_w2 = -40

    g_bias = 0
    # график номер 2 будет при первой генерации по умолчанию иметь то значение
    # b, которое выставлено в первом
    X_corrupted = X.copy()
    y_corrupted = y.copy()

    def visualize_cost_function(fixed_bias, mixing, shifting):
        """ Визуализируем поверхность целевой функции на (опционально) подпорченных
        данных и сами данные.  Портим данные мы следующим образом: сдвигаем
        категории навстречу друг другу, на величину, равную shifting Кроме того,
        меняем классы некоторых случайно выбранных примеров на противоположнее.
        Доля таких примеров задаётся переменной mixing

        Нам нужно зафиксировать bias на определённом значении, чтобы мы могли
        что-нибудь визуализировать.  Можно посмотреть, как bias влияет на форму
        целевой функции """
        xlim = (min_w1, max_w1)
        ylim = (min_w2, max_w2)
        xx = np.linspace(*xlim, num=101)
        yy = np.linspace(*ylim, num=101)
        xx, yy = np.meshgrid(xx, yy)
        points = np.stack([xx, yy], axis=2)

        # не будем портить исходные данные, будем портить их копию
        corrupted = data.copy()

        # инвертируем ответы для случайно выбранного поднабора данных
        mixed_subset = np.random.choice(range(len(corrupted)),
                                        int(mixing*len(corrupted)),
                                        replace=False)
        corrupted[mixed_subset, -1] = np.logical_not(corrupted[mixed_subset,
                                                               -1])

        # сдвинем все груши (внизу справа) на shifting наверх и влево
        pears = corrupted[:, 2] == 1
        apples = np.logical_not(pears)
        corrupted[pears, 0] -= shifting
        corrupted[pears, 1] += shifting

        # вытащим наружу испорченные данные
        global X_corrupted, y_corrupted
        X_corrupted = np.hstack((np.ones((len(corrupted), 1)), corrupted[:,
                                                                         :-1]))
        y_corrupted = corrupted[:, -1].reshape((len(corrupted), 1))

        # посчитаем значения целевой функции на наших новых данных
        calculate_weights = partial(J_by_weights, X=X_corrupted, y=y_corrupted,
                                    bias=fixed_bias)
        J_values = np.apply_along_axis(calculate_weights, -1, points)

        fig = plt.figure(figsize=(16, 5))
        # сначала 3D-график целевой функции
        ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax_1.plot_surface(xx, yy, J_values, alpha=0.5)
        print(type(surf))
        ax_1.set_xlabel("$w_1$")
        ax_1.set_ylabel("$w_2$")
        ax_1.set_zlabel("$J(w_1, w_2)$")
        ax_1.set_title("$J(w_1, w_2)$ for fixed bias = ${}$".format(fixed_bias))
        # потом плоский поточечный график повреждённых данных
        ax_2 = fig.add_subplot(1, 2, 2)
        plt.scatter(corrupted[apples][:, 0], corrupted[apples][:, 1],
                    color="red", alpha=0.7)
        plt.scatter(corrupted[pears][:, 0], corrupted[pears][:, 1],
                    color="green", alpha=0.7)
        ax_2.set_xlabel("yellowness")
        ax_2.set_ylabel("symmetry")

        plt.show()

    data = np.loadtxt("data.csv", delimiter=",")
    pears = data[:, 2] == 1
    apples = np.logical_not(pears)

    def learning_curve_for_starting_point(b, w1, w2, learning_rate=0.1):
        w = np.array([b, w1, w2]).reshape(X_corrupted.shape[1], 1)
        print(w)
        learning_rate = float(learning_rate)
        neuron = Neuron(w, activation_function=sigmoid,
                        activation_function_derivative=sigmoid_prime)

        story = [J_quadratic(neuron, X_corrupted, y_corrupted)]
        for _ in range(2000):
            neuron.SGD(X_corrupted, y_corrupted, 2, learning_rate=learning_rate,
                       max_steps=2)
            story.append(J_quadratic(neuron, X_corrupted, y_corrupted))
        plt.plot(story)
        outstr = """Learning curve.
                    Final $b={0:.3f}$, $w_1={1:.3f}, w_2={2:.3f}$"""
        plt.title(outstr.format(*neuron.w.ravel()))
        plt.ylabel("$J(w_1, w_2)$")
        plt.xlabel("Weight and bias update number")
        plt.show()

    visualize_cost_function(0, 0, 0)
    for learning_rate in [1e-4, 0.01, 0.1, 1, 10, 100]:
        learning_curve_for_starting_point(0.0, 0.0, 20.0, learning_rate)
