class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda:None

    def backward(self):
        if self.grad == 0.0:
            self.grad = 1.0

        if self._op=="+":
            self.prev[0].grad = 1.0
            self.prev[1].grad = 1.0

        if self._op=="*":
            self.prev[0].grad = self.prev[1].data
            self.prev[1].grad = self.prev[0].data

        


    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            return Value(self.data + other, (self,other), "+")

        return Value(self.data + other.data,(self,other),"+")
    
    def __mul__(self,other):
        if not isinstance(other,Value):
            return Value(self.data * other, (self,other), "*")
        return Value(self.data * other.data, (self,other),"*")
    
    def __sub__(self,other):
        if not isinstance(other,Value):
            return Value(self.data - other, (self,other), "-")
        return Value(self.data - other.data, (self,other),"-")
    
    def __pow__(self,other):
        pass


    

a = Value(4)
b = Value(3)
c = a+b
d = c + 3

print(d)
print(a.grad)