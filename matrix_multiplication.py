import numpy as np
from collections import defaultdict

row_col=[]
final_output=[]
partial_sum=[]
binary_inp=[]
binary_weights=[]
input_list=[]
weight_list=[]
actual_output=[]
matrix_mult=[]
f_sum=[]
output=[]
answer=' '
string=' '
string2=' '

partial_sums= defaultdict(list)
partial_acc= defaultdict(list)
bit_sum= defaultdict(list)


def conv_inp_to_8bit(inputs):

	binary_i= bin(inputs)
	length= len(binary_i)

	if length==10:
		binary_i= binary_i[2:: ]


	else:

		binary_i= binary_i[2:: ]

		for i in range(0,8-len(binary_i)):

			binary_i= '0'+binary_i

	return binary_i


def conv_weight_to_4bit(weights):

	binary_w= bin(weights)
	length= len(binary_w)

	if length==6:
		binary_w= binary_w[2:: ]


	else:

		binary_w= binary_w[2:: ]

		for i in range(0,6-length):

			binary_w= '0'+binary_w

	return binary_w

def addBinaryUtil(a, b): 
      
    result = ""; # Initialize result 
    s = 0;       # Initialize digit sum 
  
    # Traverse both strings  
    # starting from last characters 
    i = len(a) - 1; 
    j = len(b) - 1; 
    while (i >= 0 or j >= 0 or s == 1): 
  
        # Compute sum of last digits and carry 
        s += (ord(a[i]) - ord('0')) if(i >= 0) else 0; 
        s += (ord(b[j]) - ord('0')) if(j >= 0) else 0; 
  
        # If current digit sum is 1 or 3,  
        # add 1 to result 
        result = chr(s % 2 + ord('0')) + result; 
  
        # Compute carry 
        s //= 2; 
  
        # Move to next digits 
        i -= 1; 
        j -= 1; 
  
    return result; 
  
# function to add n binary strings 
def addBinary(list1): 
    result = ""; 
    for i in range(0,len(list1)): 
        result = addBinaryUtil(result, list1[i]); 
    return result; 


def mult_weight_with_input_bin(binary_i,binary_w,j):

	value=''
	for i in range(0,len(binary_w)):

		value= value+ str(int(binary_w[i])*int(binary_i[j]))

	value= int(value,2)

	return value

def mult_input_weights_bit(inputs,weights,bit):

	answer=' '
	partial_sum.clear()
	sum_f=0

	# #print('Length weights:', len(weights))
	# #print('Length inputs:', len(inputs))

	for i in range(0,len(inputs)):

		bin_inp=' '
		bin_inp= inputs[i]
		#bin_inp= conv_inp_to_6bit(bin_inp)
		bin_inp= conv_inp_to_8bit(bin_inp)
		##print('length of binary input is:', len(bin_inp))
		# FixMe -> How can the same range work for both inputs and weights?
		bin_weight= weights[i]
		bin_weight= conv_weight_to_4bit(bin_weight)

		res_f= mult_weight_with_input_bin(bin_inp,bin_weight,bit)
		partial_sum.append(res_f)

	#print('Partial sum is:', partial_sum)

	for m in range(0,len(partial_sum)):

		sum_f= sum_f+ partial_sum[m]

	#print('Sum_f is:', sum_f)

	end= len(partial_sum)
	#print('End is:', end)

	sum_f1 = np.round((sum_f/(end*15)), 5)

	# Since the max value can only go to 0.9375V after double boosting; hence multiplying by that 
	sum_f2= np.round((sum_f1*0.9375),5)
	#print('After mult with 0.9375 and div by 480: %s' % (sum_f2))

	# Now sum_f2 needs to be divided by the delta step (0.9375/15) to give a digital O/P 
	step = 0.9375/15
	sum_f= sum_f2//step
	#print('this is partial sum after dividing voltage with delta step: ', int(sum_f))

	binary_f= bin(int(sum_f))
	#print('Avg. sum_f in binary:', binary_f)
	#print('\n')
	length= len(binary_f)

	if length==6:

		binary_s= binary_f[2:: ]

	elif length<6:

		binary_s= binary_f[2:: ]

		for i in range(0,6-length):

			binary_s= '0'+binary_s

	elif length>6:

		binary_c = binary_f[2:: ]
		binary_s=''

		for i in range(0,4):

			binary_s= binary_s+ binary_c[i] 

	return binary_s

		
def dot_product(matrix1, matrix2):

	h,w= matrix1.shape
	d,f= matrix2.shape
	answer=' '
	element=' '
	count_value = 32

	for row in matrix2.transpose():

		c= row
		#print('Row is:', row)

		for row in matrix1:

			r= row
			t=7
			int_mult_out = r * c
			int_mult_out_sum = np.sum(int_mult_out)

			while t>=0:

				i=0
				j=0
				count=1

				while j<len(c):

					inp= r[i]
					weight= c[j]
					binary_inp.append(inp)
					binary_weights.append(weight)
					##print(binary_weights)

					if count%count_value==0:

						bit_s= mult_input_weights_bit(binary_inp,binary_weights,t)
						bit_sum[t].append(bit_s)
						##print(bit_sum[t])
						binary_inp.clear()
						binary_weights.clear()

					count=count+1
					j=j+1
					i=i+1

				#print('Length of the sum bits:', len(bit_sum[t]))
				numb= len(bit_sum[t])
				sums_bit=' '
				sums_bit= addBinary(bit_sum[t])
				#print('Sum :',sums_bit)
				part_sum= int(sums_bit,2)
				part_sum= int(part_sum/numb)
				sums_bit= bin(part_sum)
				sums_bit= sums_bit[2:: ]
				#print('Final sums_bit:', sums_bit)

				if len(sums_bit)>4:

					for u in range(0,4):
						answer= answer + sums_bit[u]
				else:
					answer= sums_bit

				if t<7:

					for i in range(0,7-t):

						answer= answer+'0'


				f_sum.append(answer)
				#print('F_sum is:', f_sum)
				answer=' '
				t=t-1

			all_bits_sum= addBinary(f_sum)
			element= all_bits_sum

			if len(element)>12:
				element= element[:12]

			elif len(element)<12:
				for g in range(0,(12-len(element))):
					element='0'+element

			element= int(element,2)
			final_output.append(element)
			element=' '
			f_sum.clear()
			bit_sum.clear()


	return final_output



def actual_matrix_mult(matrix1, matrix2):

	end = 128
	bin_exp_out=' '
	#final_mult = dot_product(matrix1,matrix2)
	golden_matmult = matrix1 @ matrix2
	##print('Final multiplication is: %s' % ',' .join(map(str, final_mult)))
	##print('Golden value is:', golden_matmult/end)

	list1=[]
	actual_output.clear()
	matrix_mult.clear()

	for j in range(0,len(golden_matmult)):
		list1.append(golden_matmult[j])


	for i in range(0,len(list1[0])):
		##print('MAC O/P number %s in binary: %s' %(i+1, bin(int(list1[0][i]//end))))
		bin_exp_out= bin(int(list1[0][i]//end))
		bin_exp_out= bin_exp_out[2:: ]

		if len(bin_exp_out)>12:
			bin_exp_out= bin_exp_out[:12]
						
		elif len(bin_exp_out)==12:
			bin_exp_out= bin_exp_out

		elif len(bin_exp_out)<12:
			for l in range(0,(12-len(bin_exp_out))):
				bin_exp_out='0'+bin_exp_out

		bin_exp_out= int(bin_exp_out,2)

		actual_output.append(bin_exp_out)

	return actual_output



