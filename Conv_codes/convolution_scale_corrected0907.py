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

import convolution_corrected0907 as cm
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt

conv_mult_obt=[]
conv_mult_actual=[]
output= np.zeros((1000,6))
output1= np.array([0,0,0,0,0,0])
diff= defaultdict(list)
scale= defaultdict(list)
filename= "diff_values_conv_27_adc_4_8bit.csv"
filename1= "values_conv.csv"
filename2= "scale_values_conv_acc27.csv"

l_val= input("Enter L:")
n_adc= input("Enter N_ADC:")
n_fin= input("Enter N_FIN:")

for i in range(0,1000):

	# inputs = np.random.randint(256, size=(5,5))
	# weights = np.random.randint(16, size=(6,5,5))

	inputs = np.random.randint(256, size=(3,9))
	weights = np.random.randint(16, size=(6,3,9))
	conv_mult_obt = cm.final_multiplication(inputs,weights,int(l_val), int(n_adc), int(n_fin))
	conv_mult_actual= cm.actual_convolution(inputs,weights,int(l_val), int(n_fin))

	print('obtained output:',conv_mult_obt)
	print('actual output:',conv_mult_actual)
	#np.savetxt(filename1, conv_mult_obt, delimiter=",")
	#np.savetxt(filename1, conv_mult_actual, delimiter="\n")
	#time.sleep(0.5)
	#print('obtained marix length:',len(conv_mult_obt))
	#print('expected matrix length:', len(conv_mult_actual))
	for s in range(0,len(conv_mult_actual)):
		ratio= conv_mult_actual[s]/conv_mult_obt[s]
		scale[i].append(np.round(ratio,3))
		differ = conv_mult_obt[s] - conv_mult_actual[s]
		diff[i].append(differ)
		

	#print(scale[i])
	a= np.array(diff[i])
	b= np.array(scale[i])
	output[i]=a
	output1= np.vstack((output1,b))

	# conv_mult_obt.clear()
	# conv_mult_actual.clear()
	
	del conv_mult_obt[:]
	del conv_mult_actual[:]

#print(scale[0])

output= output.flatten()
median = np.median(output)
output= output- median
#print(output)

##plotting
plt.title("1.0K random samples, $N_{IN}$ = 8 bits in 8 groups, $N_{W}$= 4 bits in 1 group, $N_{ADC}$= 7, L = 25 ", fontsize = 12)
plt.xlabel("$MAC_{Observed}$ - $MAC_{Expected}$", fontsize = 12)
plt.ylabel('Number of Occurances', fontsize = 16)
plt.ylim(0, 60)
#plt.xlim(-400,400)
plt.hist(output, color='blue', bins=50)
plt.grid(axis='y')
plt.grid(axis='x')
plt.savefig('hist_25_adc_7_8bit_new.png')

print('output1:', output1)
np.savetxt(filename2, output1, delimiter=",")





