class Value:
    def __init__(self, data, _children=(), _op="",gradient=0.0):
        self.data = data
        self.prev = set(_children)
        self._op = _op
        self.gradient = gradient
        self._backward = lambda:None



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
print(d.prev)