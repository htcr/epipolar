from sympy import *

def get_matrix(C):
    C_list = list()
    for i in range(3):
        row = list()
        for j in range(4):
            item = ('%s%d%d' % (C, i, j))
            row.append(item)
        C_list.append(row)
    return Matrix(C_list)

C1 = get_matrix('A')
w = Matrix([['w1'], ['w2'], ['w3'], ['1']])
x = Matrix([['x1'], ['y1'], ['1']])

result = Matrix(C1.dot(w)) - x
print(result)