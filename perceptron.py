# -*- coding: utf-8 -*-

import numpy as np


class Perceptron():
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def summator(self, inp):
        return float(self.w.T.dot(inp) + self.b)

    def activator(self, summator):
        if summator > 0:
            return 1
        else:
            return 0

    def forward_pass(self, single_input):
        return self.activator(self.summator(single_input))

    def vectorized_forward_pass(self, input_matrix):
        summator_matrix = input_matrix.dot(self.w) + self.b
        return np.vectorize(self.activator)(summator_matrix)

    def train_on_single_example(self, example, y):
        # example(m, 1)
        yh = self.forward_pass(example)
        err = y - yh
        self.w += example*err
        self.b += err
        return abs(err)

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        """ input_matrix - матрица входов размера (n, m), y - вектор правильных
        ответов размера (n, 1) (y[i] - правильный ответ на пример
        input_matrix[i]), max_steps - максимальное количество шагов.  Применяем
        train_on_single_example, пока не перестанем ошибаться или до
        умопомрачения.  Константа max_steps - наше понимание того, что считать
        умопомрачением.  """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                # int(True) = 1, int(False) = 0, так что можно не делать if
                errors += int(error)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data = np.loadtxt("data.csv", delimiter=",")
    pears = data[:, 2] == 1
    apples = np.logical_not(pears)

    def create_perceptron(m):
        w = np.random.random((m, 1))
        return Perceptron(w, 1)

    line, = plt.plot([], [], color="black", linewidth=2)

    def plot_line(coefs):
        """
        рисует разделяющую прямую, соответствующую весам, переданным в coefs =
        (weights, bias),
        где weights - ndarray формы (2, 1), bias - число
        """
        w, bias = coefs
        a, b = - w[0][0] / w[1][0], - bias / w[1][0]
        xx = np.linspace(*plt.xlim())
        line.set_data(xx, a*xx + b)

    def step_by_step_weights(p, input_matrix, y, max_steps=1e6):
        """
        обучает перцептрон последовательно на каждой строчке входных данных,
        возвращает обновлённые веса при каждом их изменении
        p - объект класса Perceptron
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = p.train_on_single_example(example, answer)
                # здесь мы упадём, если вы забыли вернуть размер ошибки из
                # train_on_single_example
                errors += abs(error)
                if error:
                    # будем обновлять положение линии только тогда, когда она
                    # изменила своё положение
                    yield p.w, p.b

        for _ in range(200):
            yield p.w, p.b

        """
        np.random.seed(1)
        fig = plt.figure()
        plt.scatter(data[apples][:, 0],
                    data[apples][:, 1],
                    color="red",
                    marker=".",
                    label="Apples")
        plt.scatter(data[pears][:, 0],
                    data[pears][:, 1],
                    color="green",
                    marker=".",
                    label="Pears")
        plt.xlabel("yellowness")
        plt.ylabel("symmetry")
        # создаём линию, которая будет показывать границу разделения
        line, = plt.plot([], [], color="black", linewidth=2)

        from matplotlib.animation import FuncAnimation

        perceptron_for_weights_line = create_perceptron(2)
        # создаём перцептрон нужной размерности со случайными весами

        from functools import partial
        # про partial почитайте на
        # https://docs.python.org/3/library/functools.html#functools.partial
        weights_ani = partial(
            step_by_step_weights,
            p=perceptron_for_weights_line,
            input_matrix=data[:, :-1],
            y=data[:, -1][:, np.newaxis]
        )

        ani = FuncAnimation(fig,
                            func=plot_line,
                            frames=weights_ani,
                            blit=False,
                            interval=10,
                            save_count=800,
                            repeat=False)
        # если Jupyter не показывает вам анимацию - раскомментируйте строчку
        # ниже и посмотрите видео
        ani.save("perceptron_seeking_for_solution.mp4", fps=50)
        plt.show()
        """
    def step_by_step_errors(p, input_matrix, y, max_steps=1e6):
        """ обучает перцептрон последовательно на каждой строчке входных
        данных, на каждом шаге обучения запоминает количество неправильно
        классифицированных примеров и возвращает список из этих количеств
        """
        def count_errors():
            return np.abs(
                p.vectorized_forward_pass(input_matrix).astype(np.int) - y
            ).sum()
        errors_list = [count_errors()]
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))

                error = p.train_on_single_example(example, answer)
                errors += error
                errors_list.append(count_errors())
        return errors_list

    perceptron_for_misclassification = create_perceptron(2)
    errors_list = step_by_step_errors(perceptron_for_misclassification,
                                      input_matrix=data[:, :-1],
                                      y=data[:, -1][:, np.newaxis])
    plt.plot(errors_list)
    plt.ylabel("Number of errors")
    plt.xlabel("Algorithm step number")

    step_by_step_errors
    plt.show()

    from numpy.linalg import norm

    def get_vector(p):
        """возвращает вектор из всех весов перцептрона, включая смещение"""
        v = np.array(list(p.w.ravel()) + [p.b])
        return v

    def step_by_step_distances(p, ideal, input_matrix, y, max_steps=1e6):
        """обучает перцептрон p и записывает каждое изменение расстояния от
        текущих весов до ideal"""
        distances = [norm(get_vector(p) - ideal)]
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))

                error = p.train_on_single_example(example, answer)
                errors += error
                if error:
                    distances.append(norm(get_vector(p) - ideal))
        return distances

    np.random.seed(42)
    init_weights = np.random.random_sample(3)
    w, b = init_weights[:-1].reshape((2, 1)), init_weights[-1]
    ideal_p = Perceptron(w.copy(), b.copy())
    ideal_p.train_until_convergence(data[:, :-1], data[:, -1][:, np.newaxis])
    ideal_weights = get_vector(ideal_p)

    new_p = Perceptron(w.copy(), b.copy())
    distances = step_by_step_distances(new_p,
                                       ideal_weights,
                                       data[:, :-1],
                                       data[:, -1][:, np.newaxis])

    plt.xlabel("Number of weight updates")
    plt.ylabel("Distance between good and current weights")
    plt.plot(distances)
    plt.show()
