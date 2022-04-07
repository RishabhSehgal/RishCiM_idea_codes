
'''
####################### convolution.py ########################
# This file implements Rish's idea of CiM HW Convolution for the Conv layers of NN
# Also calculates GOLDEN CiM Convolution for the same weights and Inputs file to compare.
# This is called internally by the "convolution_scale.py" to generate scaling factors.
# Created on: 04/19/2020
# Created by: Rishabh Sehgal, Aarushi Sehgal
# Version: rev_1.0
# Usage instructions: This file is used internally by wrapper file "convolution_scale.py"
#############################################################
'''


import numpy as np
import random
import math
from collections import defaultdict
import time

row_col=[]
conv_mult_actual=[]
conv_mult_out=[]
conv_mult_exp=[]
wishful=[]
final_output=[]
output=[]
convolve=' '


partial_sums=defaultdict(list)
filters=defaultdict(list)
partial_acc= defaultdict(list)

def conv_weight_to_4bit(weights):

	binary_w= bin(weights)
	print(binary_w)
	length= len(binary_w)
	print(length)
	print('\n')
	if length==6:
		binary_w= binary_w[2:: ]
		print(binary_w)
	else:
		binary_w= binary_w[2:: ]
		for i in range(0,6-length):
			binary_w= '0'+binary_w
		print(binary_w)

	print('\n')
	return binary_w


def conv_weight_to_4bit_array(weights):
	print(np.array(list(map(conv_weight_to_4bit_array, weights))))
	return np.array(list(map(conv_weight_to_4bit_array, weights)))


def conv_inp_to_8bit(inputs):

	binary_i= bin(inputs)
	print(binary_i)
	length= len(binary_i)
	print(length)
	print('\n')
	if length==10:
		binary_i= binary_i[2:: ]
		print(binary_i)
	else:
		binary_i= binary_i[2:: ]
		###print('binary inp before adding 0\'s',binary_i)
		for i in range(0,8-len(binary_i)):
			binary_i= '0'+binary_i
		print(binary_i)

	print('\n')
	return binary_i


def mult_weight_with_input(binary_i,binary_w,j):

	value=''
	for i in range(0,len(binary_w)):
		value= value+ str(int(binary_w[i])*int(binary_i[j]))
	print(value)
	value= int(value,2)
	print(value)
	print('\n')

	return value

def addBinaryUtil(a, b): 
      
    result = "" # Initialize result 
    s = 0       # Initialize digit sum 
    # Traverse both strings  
    # starting from last characters 
    i = len(a) - 1
    j = len(b) - 1 
    while (i >= 0 or j >= 0 or s == 1): 
        # Compute sum of last digits and carry 
        s += (ord(a[i]) - ord('0')) if(i >= 0) else 0 
        s += (ord(b[j]) - ord('0')) if(j >= 0) else 0 
        # If current digit sum is 1 or 3,  
        # add 1 to result 
        result = chr(s % 2 + ord('0')) + result
  
        # Compute carry 
        s //= 2
  
        # Move to next digits 
        i -= 1
        j -= 1
  
    return result
  
# function to add n binary strings 
def addBinary(list1): 
    result = ""
    for i in range(0,len(list1)): 
        result = addBinaryUtil(result, list1[i]) 
        print(result)
    print('\n')
    return result


def final_multiplication(inputs, weights, p, n_adc, n_fin ):

	input_list=[]
	partial_sums=defaultdict(list)
	filters=defaultdict(list)
	partial_acc= defaultdict(list)

	h,w= inputs.shape
	d,l,f= weights.shape
	answer=' '
	for i in range(0,h):
		for j in range(0,w):
			string= inputs[i][j]
			res= conv_inp_to_8bit(int(string))
			input_list.append(res)

	print(input_list)
	print('\n')

	for i in range(0,d):
		for j in range(0,l):
			for k in range(0,f):
				string2= weights[i][j][k]
				res1= conv_weight_to_4bit(int(string2))
				filters[i].append(res1)

	print(filters)
	print('\n')

	for y in range(0,d):
		## y iterates over number of filters.
		bit= 7
		for l in range(0,8):
			sum_f=0
			for k in range(0,p):
				## k iterates over all the weights
				## Iterates over all the weights; Here we have 25 weights in a row
				res_f= mult_weight_with_input(input_list[k],filters[y][k],bit)
				# print('input is:', input_list[k])
				# print('filter is:', filters[y][k])
				# print('bit is:', bit)
				# print('res_f:', res_f)
				partial_sums[l].append(res_f)

			print(partial_sums)
			print('\n')
			for m in range(0,p):
				sum_f= sum_f+ int(partial_sums[l][m])

			print(sum_f)
			print('\n')
			sum_f= int((sum_f/p)) #(removed *2 from here)
			#print('this is partial sum:',sum_f)
			binary_f= bin(sum_f)
			binary_f=binary_f[2:: ]
			#print(binary_f)
			length= len(binary_f)
			if length==n_adc:
				binary_f= binary_f
			elif length<n_adc:
				for i in range(0,n_adc-len(binary_f)):
					binary_f= '0'+binary_f
			elif length>n_adc:
				binary_f= binary_f[:n_adc]
			if l>0:
				for i in range(0,l):
					binary_f= binary_f+'0'
			# print(binary_f)
			# time.sleep(1)
			partial_acc[y].append(binary_f)
			bit=bit-1


		final_sum= addBinary(partial_acc[y])
		answer= final_sum
		if len(answer)==n_fin:
			answer= answer
		elif len(answer)>n_fin:
			answer= answer[:n_fin]
		elif len(answer)<n_fin:
			for h in range(0,(n_fin-len(answer))):
				answer='0'+answer
		#print(answer)
		answer= int(answer,2)
		output.append(answer)
		partial_sums.clear()
		answer=' '

	return output

def actual_convolution(x,y,p, n_fin):

	actual_conv = np.sum((np.sum((x * y), axis =1)), axis = 1)
	del conv_mult_actual[:]
	#conv_mult_actual.clear()

	for i in range(0,len(actual_conv)):
		convolve= bin(actual_conv[i]//p)
		convolve= str(convolve[2:: ])
		if len(convolve)==n_fin:
			convolve= convolve
		elif len(convolve)>n_fin:
			convolve= convolve[:n_fin]
		elif len(convolve)<n_fin:
			for u in range(0,(n_fin-len(convolve))):
				convolve='0'+convolve
		convolve= int(convolve,2)
		conv_mult_actual.append(convolve)
	
	return conv_mult_actual


