import copy

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

A = Matrix([
    [1,2],
    [5,8]
])

B = Matrix([
    [1,2],
    [3,4]
])

print(A.add(B))

print(A.shape())        # (2,3)

print(A.transpose())
# [
#   [1,4],
#   [2,5],
#   [3,6]
# ]