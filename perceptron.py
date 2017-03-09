# -*- coding: utf-8 -*-


class Perceptron():
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def summator(self, inp):
        return self.w.T.dot(inp) + self.b

    def forward_pass(self, summator):
        if summator > 0:
            return 1
        else:
            return 0
