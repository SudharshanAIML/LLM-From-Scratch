import math
import copy


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

class Matrix:
    def __init__(self, values):
        self.values = values

    def shape(self):
        return [len(self.values) , len(self.values[0])]

    def transpose(self):
        row, col = self.shape()
        transposed = [[self.values[j][i] for j in range(row)] for i in range(col)]
        return transposed
        

    def add(self, other):
        row1, col1 = self.shape()
        row2, col2 = other.shape()
        if(row1 != row2 or col1 != col2):
            return ("these are two differnet dimensioned matrix") 
        added = [[self.values[i][j] + other.values[i][j] for j in range(len(other.values[0]))] for i in range(len(other.values))]
        return added

class VectorMatrix:

    @staticmethod
    def dot(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    def multiply_vector(self,M , V):
        r, c = M.shape()
        sz = len(V.values)

        if(c != sz):
            return "dimensions are not correct"

        res = []
        for  row in M.values:
            res.append(self.dot(row , V.values))

        return res







M = Matrix([
    [1,2,5,7],
    [3,4,7,8],
    [9,10,11,2],
    [2,1,1,1]
])
V = Vector([4,5,6,1])

MV = VectorMatrix()
print(MV.multiply_vector(M,V))
