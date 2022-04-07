'''
##################### convolution_scale.py #####################
# This file is a wrapper file to call the convolution file, N number of times.
# That helps in finding the scaling factor between actual 11b O/P and my 11b O/P
# scaling factor = (Actual 11b output) / (My 11b output); Was this 13bit
# Outputs the scaling factors in a CSV file automatically. scale_values_conv.csv
# Created on: 04/19/2020
# Created by: Rishabh Sehgal, Aarushi Sehgal
# Version: rev_1.0
# Usage instructions: " python convolution_scale.py "
#############################################################
'''

import convolution as cm
import numpy as np
import time
from collections import defaultdict


conv_mult_obt=[]
conv_mult_actual=[]
output= np.array([0,0,0,0,0,0])

scale= defaultdict(list)
filename= "scale_values_conv.csv"
filename1= "values_conv.csv"


for i in range(0,10):

	inputs = np.random.randint(256, size=(5,5))
	weights = np.random.randint(16, size=(6,5,5))

	conv_mult_obt = cm.final_multiplication(inputs, weights)

	conv_mult_actual= cm.actual_convolution(inputs,weights)

	print('obtained output:',conv_mult_obt)
	print('actual output:',conv_mult_actual)
	np.savetxt(filename1, conv_mult_obt, delimiter=",")
	np.savetxt(filename1, conv_mult_actual, delimiter="\n")

	time.sleep(0.5)


	#print('obtained marix length:',len(conv_mult_obt))
	#print('expected matrix length:', len(conv_mult_actual))

	for s in range(0,len(conv_mult_actual)):

		#LSB= conv_mult_obt[s] - conv_mult_actual[s]
		ratio= conv_mult_actual[s]/conv_mult_obt[s]
		scale[i].append(np.round(ratio,3))

	#print(scale[i])
	a= np.array(scale[i])
	
	output=np.vstack((output,a))

	conv_mult_obt.clear()
	conv_mult_actual.clear()

#print(scale[0])

print('output:', output)
np.savetxt(filename, output, delimiter=",")






