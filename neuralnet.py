from value import Value
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        activation = sum((wi*xi for wi,xi in zip(self.w,x))) + self.b
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for neurons in self.neurons:
            ps = neurons.parameters()
            params.extend(ps)
        return params
        

class MLP:

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]
        self.loss = None

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self,xs):
        ypred = [self(x) for x in xs]
        return ypred

    def fit(self, xs,ys, lr=0.01, epochs=1000):
        for epoch in range(epochs):
            for p in self.parameters():
                p.grad = 0.0
            ypred = [self(x) for x in xs]
            self.loss = sum((yout - ygt)**2 for ygt,yout in zip(ys,ypred))
            self.loss.backward()

            for p in self.parameters():
                p.data += -lr * p.grad
        
        return self.loss.data


    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]

]
n = MLP(3, [4,4,1])
ys = [1.0,-1.0,-1.0,1.0]
n.fit(xs,ys)
print(n.predict(xs))
