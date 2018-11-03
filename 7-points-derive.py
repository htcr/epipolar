from sympy import *

matrix_list = list()
for i in range(3):
    row = list()
    for j in range(3):
        idx = i*3+j
        formula = ('a%d*x+b%d' % (idx, idx))
        row.append(formula)
    matrix_list.append(row)

M = Matrix(matrix_list)
M_det = M.det()
result = collect(expand(M_det), 'x')
print(result)

'''
b0*b4*b8 - b0*b5*b7 - b1*b3*b8 + b1*b5*b6 + b2*b3*b7 - b2*b4*b6 + x**3*(a0*a4*a8 -a0*a5*a7 - a1*a3*a8 + a1*a5*a6 + a2*a3*a7 - a2*a4*a6) + x**2*(a0*a4*b8 - a0*a5*b7 - a0*a7*b5 + a0*a8*b4 - a1*a3*b8 + a1*a5*b6 + a1*a6*b5 - a1*a8*b3 + a2*a3*b7 - a2*a4*b6 - a2*a6*b4 + a2*a7*b3 + a3*a7*b2 - a3*a8*b1 - a4*a6*b2 + a4*a8*b0 + a5*a6*b1 -a5*a7*b0) + x*(a0*b4*b8 - a0*b5*b7 - a1*b3*b8 + a1*b5*b6 + a2*b3*b7 - a2*b4*b6 - a3*b1*b8 + a3*b2*b7 + a4*b0*b8 - a4*b2*b6 - a5*b0*b7 + a5*b1*b6 + a6*b1*b5 - a6*b2*b4 - a7*b0*b5 + a7*b2*b3 + a8*b0*b4 - a8*b1*b3)

'''


# 3-degree:
'''
(a0*a4*a8 -a0*a5*a7 - a1*a3*a8 + a1*a5*a6 + a2*a3*a7 - a2*a4*a6)
'''

# 2-degree:
'''
(a0*a4*b8 - a0*a5*b7 - a0*a7*b5 + a0*a8*b4 - a1*a3*b8 + a1*a5*b6 + a1*a6*b5 - a1*a8*b3 + a2*a3*b7 - a2*a4*b6 - a2*a6*b4 + a2*a7*b3 + a3*a7*b2 - a3*a8*b1 - a4*a6*b2 + a4*a8*b0 + a5*a6*b1 -a5*a7*b0)
'''

# 1-degree:
'''
(a0*b4*b8 - a0*b5*b7 - a1*b3*b8 + a1*b5*b6 + a2*b3*b7 - a2*b4*b6 - a3*b1*b8 + a3*b2*b7 + a4*b0*b8 - a4*b2*b6 - a5*b0*b7 + a5*b1*b6 + a6*b1*b5 - a6*b2*b4 - a7*b0*b5 + a7*b2*b3 + a8*b0*b4 - a8*b1*b3)
'''
# 0-degree:
'''
(b0*b4*b8 - b0*b5*b7 - b1*b3*b8 + b1*b5*b6 + b2*b3*b7 - b2*b4*b6)
'''