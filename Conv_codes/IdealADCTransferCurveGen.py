import matplotlib.pyplot as plt
import pandas
import numpy as np

######### Function Definition ##############
def gen_ideal_adc_trans_curve(
        vin_min, vin_max, v_sample_ideal_arr, res_bits):
    # vin_los, vin_high : full scale input voltages in mV
    # v_sample_ideal_arr: 1D array of sampled voltages in the sweep plot in mV
    # res_bits          : ADC resolution in # of bits

    num_samples = v_sample_ideal_arr.shape[0]
    print("num_samples = {}".format(num_samples))

    full_scale = vin_max - vin_min     # in units of mV
    num_codes = 2 ** res_bits
    res_v = full_scale / num_codes      # LSB resolution in mV

    # Create bins
    code_bins_arr = np.zeros((num_codes, 2))    # col0: lower voltage of output code
                                                # col1: higher voltage of output code
    for i in range(num_codes):
        code_bins_arr[i, :] = [vin_min + i * res_v, vin_min + (i + 1) * res_v]

    # find the code bin each v_sample in v_sample_arr belongs to
    d_sample_ideal_arr = np.empty(v_sample_ideal_arr.shape)
    for sample_idx in range(num_samples):
        for i in range(num_codes):
            if code_bins_arr[i, 0] <= v_sample_ideal_arr[sample_idx] < code_bins_arr[i, 1]:
                d_sample_ideal_arr[sample_idx] = i
                break
            elif v_sample_ideal_arr[sample_idx] == vin_max:
                d_sample_ideal_arr[sample_idx] = num_codes-1
                break

    return (d_sample_ideal_arr, res_v)

def correct_offset_gain_error(
        vin_min_ideal, vin_max_ideal, res_bits,
        vin_min_act, vin_max_act,
        d_sample_act,
):
    # vin_min_ideal = vin_min + res_v/2    in mV
    # vin_max_ideal = vin_max - res_v/2    in mV
    # vin_min_act is the last vin that corresponds to a digital code of 0
    # vin_max_act is the first vin that corresponds to a digital code fo 2**res_bits-1

    m_act   = float((2**res_bits-1)) / (float(vin_max_act)   - float(vin_min_act)  )
    m_ideal = float((2**res_bits-1)) / (float(vin_max_ideal) - float(vin_min_ideal))

    # Assume d_act   = m_act  *(vin - vin_min_act)
    #        d_ideal = m_ideal*(vin - vin_min_ideal)
    #        d_oc is the offset corrected version of d_act:
    #        d_oc    = m_act  *(vin - vin_min_ideal)

    # offset correction
    # d_oc = d_act + m_act*(vin_min_act - vin_min_ideal)
    d_sample_oc = d_sample_act + m_act*(vin_min_act - vin_min_ideal)

    # gain correction
    # to get form d_oc to d_ideal:
    # d_ideal = d_oc / m_act * m_ideal
    d_sample_gc = d_sample_oc / m_act * m_ideal

    return(np.rint(d_sample_gc))

def correct_offset_gain_error_line_fit(
        vin_min_ideal, vin_max_ideal, res_bits,
        vin_start_act, vin_end_act,
        v_sample_act, d_sample_act
):
    # vin_min_ideal = vin_min + res_v/2    in mV
    # vin_max_ideal = vin_max - res_v/2    in mV
    # vin_start_act is the start voltage that best-line fit on actual curve is conducted in mV
    # vin_end_act   is the end   voltage that best-line fit on actual curve is conducted in mV

    start_idx = np.where(v_sample_act == vin_start_act)[0][0]
    end_idx = np.where(v_sample_act == vin_end_act)[0][0]

    # d_ideal = m_ideal * vin + b_ideal
    m_ideal = float((2**res_bits-1)) / (float(vin_max_ideal) - float(vin_min_ideal))
    b_ideal = -m_ideal * (vin_min_ideal + res_v / 2)

    # d_act = m_act * vin + b_act
    tmp = np.polyfit(v_sample_act[start_idx:end_idx+1], d_sample_act[start_idx:end_idx+1], 1)
    m_act = tmp[0]
    b_act = tmp[1]
    # offset correction
    d_sample_corrected = (d_sample_act - b_act) / m_act * m_ideal + b_ideal

    return (np.clip(np.rint(d_sample_corrected), 0, 2**res_bits-1))


#########################################

################Script Start#############

##### set ideal parameters####

vin_min = 0
vin_max = 465
res_bits = 4
step = 5

# vin_min = 300
# vin_max = 900
# res_bits = 7
# step = 2
##############################

# Create ideal ADC transfer curve
v_sample_ideal_arr = np.arange(vin_min, vin_max + step, step)
(d_sample_ideal_arr, res_v) = gen_ideal_adc_trans_curve(vin_min, vin_max, v_sample_ideal_arr, res_bits)


#####Read in the measured ADC transfer curve#####
#####and correct offset and gain#################
df = pandas.read_csv("../../../tsmc65n_simulation/CiSRAM_ADC_txcurve/vin_sweep_run10_91pnts_C_SET_VDD1p0_OTAbiasp0p650_390f_tunedVrefs_06092021_process.csv", delimiter=' ')
# vin_min_act is the last vin that corresponds to a digital code of 0
# vin_max_act is the first vin that corresponds to a digital code fo 2**res_bits-1
# vin_start_act is the start voltage that best-line fit on actual curve is conducted in mV
# vin_end_act   is the end   voltage that best-line fit on actual curve is conducted in mV
vin_min_act = 30
vin_max_act = 475
vin_start_act = 0
vin_end_act   = 465
df_arr = np.array(df.values.tolist())
v_sample_arr = df_arr[:, 0] * 1000
num_samples2 = v_sample_arr.shape[0]
print("num_samples2 = {}".format(num_samples2))

# df = pandas.read_csv("../Papers/ForISSCCTable/sch_sweep_390m_950m_100MHz.csv")
# vin_min_act = 390
# vin_max_act = 934
# vin_start_act = 420
# vin_end_act = 850
# df_arr = np.array(df.values.tolist())
# v_sample_arr = df_arr[:, 0] * 1000
##############################################

d_sample_arr = df_arr[:, 1]
d_sample_corrected = correct_offset_gain_error_line_fit(
                vin_min_ideal=vin_min + res_v/2,
                vin_max_ideal=vin_max - res_v/2,
                res_bits=res_bits,
                vin_start_act=vin_start_act,
                vin_end_act=vin_end_act,
                v_sample_act=v_sample_arr,
                d_sample_act=d_sample_arr
)


combined_array = np.column_stack((v_sample_ideal_arr, d_sample_ideal_arr, v_sample_arr, d_sample_arr, d_sample_corrected))
# # Export to spreadsheet
np.savetxt("../../../tsmc65n_simulation/ForISSCCTable/corrected_ver3.csv", d_sample_corrected.astype(int), delimiter=",")
np.savetxt("../../../tsmc65n_simulation/ForISSCCTable/ideal_ver3.csv", d_sample_ideal_arr.astype(int), delimiter=",")
np.savetxt("../../../tsmc65n_simulation/ForISSCCTable/combined_ver3.csv", combined_array, delimiter=",")

## Read measured
df2 = pandas.read_csv("../../../tsmc65n_simulation/CiSRAM_ADC_txcurve/measured.csv", delimiter=' ')
df_arr2 = np.array(df2.values.tolist())

# plot
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig0, ax0 = plt.subplots(figsize=(13,13))
plt.title("Measured ADC transfer curves without dynamic range adjustment", fontsize = 28)
ax0.plot(v_sample_ideal_arr, d_sample_ideal_arr, label="ideal")
ax0.plot(v_sample_arr, d_sample_corrected, label="simulated")
ax0.plot(df2.Vmav, df2.measured_output, 'bs', label="measured")
#ax0.plot(x='Vmav', y='measured_output', style='o')

#df.plot(style=['o','rx'])

ax0.legend()
plt.grid(axis='y')
plt.grid(axis='x')
plt.xlabel("$V_{MAV}$ (mV)", fontsize = 24)
plt.ylabel('Quantization output', fontsize = 24)
plt.ylim(0, 16)
plt.xlim(0, 490)
#plt.show()
plt.savefig('../../../tsmc65n_simulation/ForISSCCTable/normalrange_adc_txcurve_ver3.png')
print("End of Program.")
#########################################
