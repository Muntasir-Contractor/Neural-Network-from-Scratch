import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda:None

    def backward(self):
        topo = []
        visited = set()

        """ Even though theres already an implied tree relation with each Value and children, to avoid recomputing
        gradient multiple times for one node, we must create a flattened list of nodes in topological order. Then
        we can run a _backwards function call on the reverse of the topological sorted list, beginning at the last
        (output node), going all the way to the front layer computing the gradients in respect to the output node.
        """
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    build_topo(child)

                topo.append(node)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            if node._backward:
                node._backward()
        




    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self+other
    
    def __rmul__(self,other):
        return self*other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data-other.data,(self,other),"-")
        def _backward():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self,other):
        out = Value(self.data**other, (self,),f"**{other}")
        def _backward():
            self.grad += other*(self.data**(other-1))* out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2)*out.grad

        out._backward = _backward
        return out


if __name__ == "__main__":
    

    a = Value(4)
    b = Value(3)
    d = Value(-2)
    c = a+b
    e = c * d
    f = e**2




    f.backward()
    print(f.grad)
    print(e.grad)
    print(c.grad)
    print(d.grad)
    print(b.grad)
    print(a.grad)