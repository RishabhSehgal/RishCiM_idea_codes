import numpy as np
import pandas
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.family"] = "arial"


######### Function Definition ##############
def gen_transition_levels(v_sample, d_sample, res_bits, res_v):
    # v_sample is 1D numpy array; element values are increasing; in mV
    # d_sample is 1D numpy array dtype=int; d_sample[0] should be the smallest code
    # res_v is the LSB measured in mV
    # the sampling step in v_sample should be < res_v

    code_min = np.amin(d_sample)
    code_max = np.amax(d_sample)

    # Transition level array T[k] is defined T[code_min+1] and T[code_max] (inclusive)
    T = np.empty(2**res_bits)
    T[:] = np.NaN

    current_code = d_sample[0]
    for code_idx in range(1, d_sample.size):
        if d_sample[code_idx] == current_code + 1:
            T[current_code+1] = v_sample[code_idx] / res_v
            current_code += 1
        elif d_sample[code_idx] > current_code + 1:
            num_missing_code = d_sample[code_idx] - current_code - 1
            T[(current_code+1):(current_code+1)+num_missing_code+1] = \
                np.full(num_missing_code+1, v_sample[code_idx] / res_v)
            current_code = d_sample[code_idx]
        elif d_sample[code_idx] == current_code:
            pass
        elif d_sample[code_idx] < current_code:
            print("Error: d_sample[{}] = {} is smaller than the previous code d_sample[{}] = {}!"
                  .format(code_idx, d_sample[code_idx], code_idx-1, d_sample[code_idx-1]))
            exit(1)
        else:
            print("Uncaught case: current_code = {}; d_sample[{}] = {}"
                  .format(current_code, code_idx, d_sample[code_idx]))
            exit(2)
    return T, code_min, code_max

def gen_code_width(T, code_min, code_max):
    # T is the 1D array that had transition levels T[code_min+1] to T[code_max] defined in units of LSB
    # code_min is the smallest digital code appeared in sweep after correction
    # code_max is the biggest  digital code appeared in sweep after correction

    # W is defined between W[code_min+1] and W[code_max-1]
    W = np.empty(T.shape)
    W[:] = np.NaN
    for code in range(code_min+1, code_max):
        W[code] = T[code+1] - T[code]
    W_avg = (T[code_max] - T[code_min+1]) / (code_max-code_min-1)
    return W, W_avg

def gen_DNL(W, W_avg, code_min, code_max):
    # W is a 1D array code width that is defined between W[code_min+1] and W[code_max-1] in unit of LSB
    # W_avg is the average code width in unit of LSB
    # code_min is the smallest digital code appeared in sweep after correction
    # code_max is the biggest  digital code appeared in sweep after correction

    # DNL is defined between DNL[code_min+1]  to DNL[code_max-1]
    DNL = np.empty(W.shape)
    DNL[:] = np.NaN
    for code in range(code_min+1, code_max):
        DNL[code] = (W[code] - W_avg) / W_avg
    return DNL

def gen_INL(DNL, code_min, code_max):
    # DNL is a 1D array that is defined between DNL[code_min+1] and DNL[code_max-1]
    # code_min is the smallest digital code appeared in sweep after correction
    # code_max is the biggest  digital code appeared in sweep after correction

    # INL is defined between INL[code_min+1] and INL[code_max]
    # INL[code_min+1] is default to 0
    # INL[code_max] should work out to be 0 in the end
    INL = np.empty(DNL.shape)
    INL[:] = np.NaN
    INL[code_min+1] = 0
    for code in range(code_min+2, code_max+1):
        INL[code] = np.sum(DNL[code_min+1:code])
    return INL
###############################################

##################Script Start##################

##########Get ideal res_v#############
vin_min_ideal = 0
vin_max_ideal = 465
res_bits = 4
res_v = (vin_max_ideal - vin_min_ideal) / 2**res_bits
#
# vin_min_ideal = 300
# vin_max_ideal = 900
# res_bits = 7
# res_v = (vin_max_ideal - vin_min_ideal) / 2**res_bits
#######################################

##########load corrected curve#############
df = pandas.read_csv("../../../tsmc65n_simulation/ForISSCCTable/corrected_ver1.csv", delimiter=' ')
df_arr = np.array(df.values.tolist())
v_sample = df_arr[:, 0] * 1000
d_sample_corrected = df_arr[:, 3].astype(int)

# df = pandas.read_csv("../Papers/ForISSCCTable/sch_sweep_390m_950m_100MHz.csv")
# df_arr = np.array(df.values.tolist())
# v_sample = df_arr[:, 0] * 1000
# d_sample_corrected = df_arr[:, 3].astype(int)
##########################################


T, code_min, code_max = gen_transition_levels(
                                v_sample=v_sample,
                                d_sample=d_sample_corrected,
                                res_bits=res_bits,
                                res_v=res_v)
W, W_avg = gen_code_width(T=T, code_min=code_min, code_max=code_max)
DNL = gen_DNL(W=W, W_avg=W_avg, code_min=code_min, code_max=code_max)
INL = gen_INL(DNL=DNL, code_min=code_min, code_max=code_max)
print(DNL)
print(INL)
# np.savetxt("../Papers/ForISSCCTable/DNL.csv", DNL, delimiter=",")
# np.savetxt("../Papers/ForISSCCTable/INL.csv", INL, delimiter=",")


######################### plot ##############################
fig_DNL, ax_DNL = plt.subplots(figsize=(12, 3))
ax_DNL.set_ylim(-1, 1.5)

# # for clipOFF
# # stem_container_DNL = ax_DNL.stem(np.arange(code_min+1, code_max), DNL[code_min+1:code_max],
# #                                  linefmt='k-', markerfmt=" ", basefmt="k-")
# line_DNL = ax_DNL.plot(np.arange(code_min+1, code_max), DNL[code_min+1:code_max], linestyle='-',
#                        color='black', linewidth=2)
# ax_DNL.set_xlim(0, 2**res_bits)

# for clipON
# stem_container_DNL = ax_DNL.stem(np.arange(code_min+1, code_max)+64, DNL[code_min+1:code_max],
#                                  linefmt='k-', markerfmt=" ", basefmt="k-")
line_DNL = ax_DNL.plot(np.arange(code_min+1, code_max)+64, DNL[code_min+1:code_max], linestyle='-',
                       color='black', linewidth=2)
ax_DNL.set_xlim(0, 2**(res_bits+1))

ax_DNL.set_xlabel('Code', fontsize=21, fontWeight='bold')
ax_DNL.set_ylabel("DNL(LSB)", fontsize=21, fontweight='bold')
ax_DNL.grid(which='major', axis='both', linestyle="--")
ax_DNL.set_xticks(np.arange(0, 257, 32))
line_zero = ax_DNL.plot(np.arange(0,257), np.zeros(257), linestyle='-', linewidth=1, color='black')

for tick in ax_DNL.xaxis.get_major_ticks():
    tick.label1.set_fontsize(17)
    tick.label1.set_fontweight('bold')
for tick in ax_DNL.yaxis.get_major_ticks():
    tick.label1.set_fontsize(17)
    tick.label1.set_fontweight('bold')

fig_INL, ax_INL = plt.subplots(figsize=(12, 3))
# # for clipOFF
# # stem_container_INL = ax_INL.stem(np.arange(code_min+1, code_max+1), INL[code_min+1:code_max+1],
# #                                  linefmt='k-', markerfmt=" ", basefmt="k-")
# line_INL = ax_INL.plot(np.arange(code_min+1, code_max+1), INL[code_min+1:code_max+1], linestyle='-',
#                        color='black', linewidth=2)
# ax_INL.set_xlim(0, 2**res_bits)

# for clipON
# stem_container_INL = ax_INL.stem(np.arange(code_min+1, code_max+1)+64, INL[code_min+1:code_max+1],
#                                  linefmt='k-', markerfmt=" ", basefmt="k-")
line_INL = ax_INL.plot(np.arange(code_min+1, code_max+1)+64, INL[code_min+1:code_max+1], linestyle='-',
                       color='black', linewidth=2)
ax_INL.set_xlim(0, 2**(res_bits+1))

ax_INL.set_xlabel('Code', fontsize=21, fontWeight='bold')
ax_INL.set_ylabel("INL(LSB)", fontsize=21, fontweight='bold')
ax_INL.grid(which='major', axis='both', linestyle="--")
ax_INL.set_xticks(np.arange(0, 257, 32))
line_zero = ax_INL.plot(np.arange(0,257), np.zeros(257), linestyle='-', linewidth=1, color='black')

for tick in ax_INL.xaxis.get_major_ticks():
    tick.label1.set_fontsize(17)
    tick.label1.set_fontweight('bold')
for tick in ax_INL.yaxis.get_major_ticks():
    tick.label1.set_fontsize(17)
    tick.label1.set_fontweight('bold')
#######################################################
# # for clipOFF
fig_DNL.savefig("../../../tsmc65n_simulation/ForISSCCTable/DNL_sim2_clipOFF.png", bbox_inches='tight')
fig_INL.savefig("../../../tsmc65n_simulation/ForISSCCTable/INL_sim2_clipOFF.png", bbox_inches='tight')

# for clipON
# fig_DNL.savefig("../Papers/ForISSCCTable/DNL_sim2_clipON.png", bbox_inches='tight')
# fig_INL.savefig("../Papers/ForISSCCTable/INL_sim2_clipON.png", bbox_inches='tight')

plt.show()

###############################################
