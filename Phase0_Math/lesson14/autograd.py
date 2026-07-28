class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward 
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data , (self,other), '*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        topo = []
        vis = set()
        def _topo(V):
            if V not in vis:
                vis.add(V)
                for child in V._prev:
                    _topo(child)
                topo.append(V)
        _topo(self)

        for v in reversed(topo):
            v._backward()


a = Value(2)
b = Value(3)

c = a * b
d = c + a



#this is loss.backward()
d.backward()

print(a.grad)
print(b.grad)

# a=2 g=1+3=4
# b=3 g=2
# c=6 g=1
# d=8 g=1
