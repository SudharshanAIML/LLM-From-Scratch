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
    def cosine_similarity(self, other):
        dt  = self.dot(other)
        mg = self.magnitude() * other.magnitude()
        if( mg != 0):
            return dt/mg
        return ("Cosine similarity is undefined for zero vectors.")



if __name__ == "__main__":
    a = Vector([1, 0])
    b = Vector([10, 0])
    c = Vector([0, 0])
    d = Vector([-1, 0])

    print(a.cosine_similarity(b))  # ≈ 1.0
    print(a.cosine_similarity(c))  # ≈ 0.0
    print(a.cosine_similarity(d))  # ≈ -1.0
