import matrix_multiplication as mm
import numpy as np
from collections import defaultdict

matrix_mult_obtained=[]
matrix_mult_actual=[]
answer=' '
list1=[]
output= np.array([0,0,0])
#scale=[]

scale= defaultdict(list)
#rows= defaultdict(list)
filename= "scale_values.csv"

for i in range(0,100):

	req_out=' '

	matrix1 = np.random.randint(255, size=(1, 128))
	##print('Matrix 1 is:', matrix1)
	matrix2 = np.random.randint(15, size=(128, 3))
	##print('Matrix 2 is:', matrix2)

	matrix_mult_obtained= mm.dot_product(matrix1, matrix2)
	#print(matrix_mult_obtained)

	matrix_mult_actual= mm.actual_matrix_mult(matrix1,matrix2)

	#print(matrix_mult_actual)
	#print('actual marix length:',len(matrix_mult_actual))
	#print('expected matrix length:', len(matrix_mult_exp))

	for s in range(0,len(matrix_mult_actual)):

		ratio= matrix_mult_actual[s]/matrix_mult_obtained[s]
		scale[i].append(np.round(ratio,3))

	a= np.array(scale[i])
	
	output=np.vstack((output,a))

	matrix_mult_obtained.clear()
	matrix_mult_actual.clear()

np.savetxt(filename, output, delimiter=",")







