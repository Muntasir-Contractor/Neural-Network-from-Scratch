from value import Value
import random
import numpy as np

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        activation = sum((wi*xi for wi,xi in zip(self.w,x))) + self.b
        out = activation.tanh()
        print(out)
        return out

class Layer:
    def __init__(self, nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs


x = [2.0,3.0]
n = Layer(2,3)
n(x)

