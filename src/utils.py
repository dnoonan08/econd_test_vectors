import numpy as np

def quickPacketSizeCalculator(packets, adcThreshold, adcm1Threshold, kappa, cm_lambda, cm_beta, ce):

    cm = (packets[:,1]&0x3ff) + ((packets[:,1]>>10)&0x3ff)
    cm_avg = (cm.sum(axis=-1)*2731)>>16

    toa   = packets[:,2:-1] & 0x3ff
    adc   = (packets[:,2:-1]>>10) & 0x3ff
    adcm1 = (packets[:,2:-1]>>20) & 0x3ff

    passADC = ((adc+ce)>(adcThreshold + ((kappa*adcm1)>>5) + ((cm_lambda*cm_avg)>>5).reshape(-1,1,1)))
    passTOA = toa>0
    passADCm1 = (adcm1>(((8*adcm1Threshold)>>5) + (cm_beta*cm_avg)>>5).reshape(-1,1,1))

    elink_payload = np.ceil(np.where(passADC,
                                     16 + 8*passADCm1 + 8*passTOA,
                                     0).sum(axis=1)/32)
    elink_payload = np.where(elink_payload>0,
                             elink_payload+2,
                             1)

    ld_size = elink_payload[:,:6].sum(axis=-1)+3
    hd_size = elink_payload[:,:12].sum(axis=-1)+3

    return ld_size,hd_size
