import numpy as np
import tqdm
from src.packetGenerator import generatePackets

#    "packets=generatePackets()\n",
#    "packets=packets.reshape(-1,12)\n",
#    "##np.savetxt('exampleData/testVectorInputs_Random.csv',packets,delimiter=',',fmt='%d')"

np.random.seed(1234)

def quickPacketSizeCalculator(packets, adcThreshold, adcm1Threshold, kappa, cm_lambda, cm_beta, NZS=False):

    # Note that the cm_avg calculation may need to be modified if you are using
    # a different number of input channels (e.g. for ECON-D on an HD wagon
    # instead of an LD CM).  The 5461 factor is only appropriate for an LD CM
    # in the default CM configuration.
    cm = (packets[:,1]&0x3ff) + ((packets[:,1]>>10)&0x3ff)
    cm_avg = (cm[:,:6].sum(axis=-1)*5461)>>16

    toa   = packets[:,2:-1] & 0x3ff
    adc   = (packets[:,2:-1]>>10) & 0x3ff
    adcm1 = (packets[:,2:-1]>>20) & 0x3ff
    tp    = (packets[:,2:-1]>>30) & 0x1
    tc    = (packets[:,2:-1]>>31) & 0x1

    passADC = (adc > (adcThreshold + ((kappa*adcm1)>>5) + ((cm_lambda*cm_avg)>>5).reshape(-1,1,1))) | NZS
    passTOA = (toa > 0)
    passADCm1 = (adcm1>(((8*adcm1Threshold)) + (cm_beta*cm_avg)>>5).reshape(-1,1,1)) | NZS
    
    channel_data_size = (((tc == 0) & (tp == 0) &  passADC &  passADCm1 &  passTOA) * 32 +
                         ((tc == 0) & (tp == 0) &  passADC & ~passADCm1 &  passTOA) * 24 +
                         ((tc == 0) & (tp == 0) &  passADC & ~passADCm1 & ~passTOA) * 16 +
                         ((tc == 0) & (tp == 0) &  passADC &  passADCm1 & ~passTOA) * 24 +
                         ((tc == 0) & (tp == 1)) * 24 +
                         ((tc == 1) & (tp == 1)) * 32 +
                         ((tc == 1) & (tp == 0)) * 32)

    raw_elink_size = np.ceil(channel_data_size.sum(axis=1) / 32).astype(int)
    elink_size = raw_elink_size + 2
    elink_size[raw_elink_size == 0] = 1

    return elink_size[:,:6].sum(axis=1)+4

################################################################################
# First do the 2 eTX scan points
#
# Start from 15% occupancy, and go up to 85%.
# Use the ZS settings from the CM pytest code.

zs_lambda = 0x08 # cm_lambda
zs_kappa = 0x08 # kappa
zs_c_i = 0xa0 # adcThreshold

zs_beta_m = 0x04 # cm_beta
zs_c_i_m = 0 # adcm1Threshold

targets = np.around(np.arange(0.15, 0.85+1e-6, 0.01), 2)
adcScales = np.array([[ 32.8       ],
       [ 35.08857143],
       [ 37.10714286],
       [ 39.33571429],
       [ 40.86828571],
       [ 43.34285714],
       [ 44.93142857],
       [ 46.76      ],
       [ 48.66857143],
       [ 50.35714286],
       [ 53.28571429],
       [ 55.47428571],
       [ 57.74285714],
       [ 59.09142857],
       [ 62.992     ],
       [ 63.92857143],
       [ 66.90714286],
       [ 70.38571429],
       [ 72.29428571],
       [ 74.68285714],
       [ 78.13142857],
       [ 79.78      ],
       [ 81.60857143],
       [ 85.53714286],
       [ 88.32571429],
       [ 91.67428571],
       [ 91.90285714],
       [ 97.27142857],
       [101.48      ],
       [101.26857143],
       [105.25714286],
       [108.34571429],
       [111.33428571],
       [114.46285714],
       [117.16142857],
       [120.23      ],
       [123.72857143],
       [127.09714286],
       [128.36571429],
       [131.37428571],
       [134.54285714],
       [136.76942857],
       [140.184     ],
       [143.10857143],
       [147.63714286],
       [149.56571429],
       [153.79828571],
       [157.96285714],
       [159.05142857],
       [164.14      ],
       [167.96857143],
       [172.93714286],
       [176.20771429],
       [178.25428571],
       [181.02885714],
       [185.97142857],
       [190.156     ],
       [195.16857143],
       [196.51314286],
       [200.22571429],
       [205.37428571],
       [211.28285714],
       [216.36742857],
       [220.48      ],
       [222.08857143],
       [227.15714286],
       [234.18571429],
       [239.69428571],
       [240.12285714],
       [249.41142857],
       [250.2       ]])


N, p = adcScales.shape
for scan_N in range(N):
    for scan_p in range(p):
        np.random.seed(1234)
        t = (targets[scan_N] - 0.15) / (0.85 - 0.15)
        packets = generatePackets(
            nEvents = 67,
            tcRate = 0.01 * t,
            tpRate = 0.04 * t,
            tctp10Rate = 0.01 * t,
            adcScale = adcScales[scan_N,scan_p],
            adcm1Scale = 10 + 210 * t,
            cmScale = 150 * t,
        )[0]
        NZS = np.zeros((67,1,1), dtype='bool')
        NZS[15] = True
        print(np.sum(quickPacketSizeCalculator(packets, zs_c_i, zs_c_i_m, zs_kappa, zs_lambda, zs_beta_m, NZS))*2, end=" ")
        packets = packets.reshape(-1, 12)
        #print(f'exampleData/testVectorInputs_2eTX_occ{int(100*targets[scan_N] + 0.0001):02d}.csv')
        np.savetxt(f'exampleData/testVectorInputs_2eTX_occ{int(100*targets[scan_N] + 0.0001):02d}.csv', packets, delimiter=',', fmt='%d')


################################################################################
# Now do the 1 eTX points
#
# Here we cannot reach 15% occupancy because of the NZS, so we start from 25%.
# We also have different ZS settings, which are just inherited from earlier CM
# pytest code.


targets = np.around(np.arange(0.25, 0.85+1e-6, 0.01), 2)
adcScales = np.array([[ 0.1618    ],
       [ 0.17653333],
       [ 0.19266667],
       [ 0.2412    ],
       [ 0.29213333],
       [ 0.37166667],
       [ 0.4552    ],
       [ 0.61053333],
       [ 0.74066667],
       [ 0.812     ],
       [ 0.86933333],
       [ 1.04886667],
       [ 1.186     ],
       [ 1.36213333],
       [ 1.49666667],
       [ 1.635     ],
       [ 1.64713333],
       [ 1.88046667],
       [ 2.202     ],
       [ 2.35993333],
       [ 2.54066667],
       [ 2.505     ],
       [ 2.81333333],
       [ 3.20166667],
       [ 3.386     ],
       [ 3.60373333],
       [ 3.86666667],
       [ 3.9968    ],
       [ 4.11533333],
       [ 4.44266667],
       [ 4.57      ],
       [ 4.97833333],
       [ 5.21466667],
       [ 5.175     ],
       [ 5.59333333],
       [ 5.76146667],
       [ 6.18      ],
       [ 6.41433333],
       [ 7.11566667],
       [ 7.6054    ],
       [ 7.64753333],
       [ 7.44566667],
       [ 8.73      ],
       [ 9.17833333],
       [ 9.10566667],
       [ 9.3716    ],
       [ 9.16333333],
       [10.04566667],
       [11.17      ],
       [11.23433333],
       [10.78666667],
       [11.875     ],
       [11.96933333],
       [12.27066667],
       [13.31      ],
       [13.53833333],
       [13.75666667],
       [13.895     ],
       [14.26533333],
       [15.20566667],
       [15.13      ]])

zs_c_i = 0
zs_c_i_m = 0
zs_kappa = 0x03
zs_lambda = 0x05
zs_beta_m = 0x05

print()
N, p = adcScales.shape
assert targets.shape[0] == N
for scan_N in range(N):
    for scan_p in range(p):
        np.random.seed(1234)
        t = (targets[scan_N] - 0.25) / (0.85 - 0.25)
        packets = generatePackets(
            nEvents = 67,
            tcRate = 0.01 * t,
            tpRate = 0.04 * t,
            tctp10Rate = 0.01 * t,
            adcScale = adcScales[scan_N,scan_p],
            adcm1Scale = 10 + 210 * t,
            cmScale = 150 * t,
        )[0]
        NZS = np.zeros((67,1,1), dtype='bool')
        NZS[15] = True
        print(np.sum(quickPacketSizeCalculator(packets, zs_c_i, zs_c_i_m, zs_kappa, zs_lambda, zs_beta_m, NZS))*2, end=" ")
        packets = packets.reshape(-1, 12)
        np.savetxt(f'exampleData/testVectorInputs_1eTX_occ{int(100*targets[scan_N] + 0.0001):02d}.csv', packets, delimiter=',', fmt='%d')

