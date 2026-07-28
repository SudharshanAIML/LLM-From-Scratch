import math

class Vector:
    def __init__(self, values):
        self.values = values

    def add(self, other):
        result = [a + b for a, b in zip(self.values, other.values)]
        return Vector(result)

    def subtract(self, other):
        result = [a - b for a, b in zip(self.values, other.values)]
        return Vector(result)

    def dot(self, other):
        return sum(a * b for a, b in zip(self.values, other.values))

    def magnitude(self):
        return math.sqrt(sum(x ** 2 for x in self.values))

if __name__ == "__main__":
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    print(a.add(b).values)        # [5, 7, 9]
    print(a.subtract(b).values)   # [-3, -3, -3]
    print(a.dot(b))               # 32
    print(round(a.magnitude(), 3))# 3.742