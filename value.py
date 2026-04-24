class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.prev = set(_children)
        self._op = _op


    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data,(self,other),"+")
    
    def __mul__(self,other):
        return Value(self.data * other.data, (self,other),"*")
    

a = Value(4)
b = Value(3)
c = a+b
d = a*b
e = c+d

print(c)
print(c.prev)
print(c._op)
print(e.prev)