import numpy as np

def generatePackets(nEvents=100,
                    tcRate=0.01,
                    tpRate=0.04,
                    tctp10Rate=0.01,
                    adcScale=220,
                    adcExponential=True,
                    adcWidth=40,
                    TOAcutoff=500,
                    cmScale=150,
                    cmExponential=True,
                    cmWidth=50
                   ):

    tc=(np.random.random(37*12*nEvents)<tcRate).astype(int)
    tp=(np.random.random(37*12*nEvents)<tpRate).astype(int)

    #tp should be 1 when tc=1, since TcTp=10 is invalid
    # but we set this to happen 1 percent of the time to exercise it
    tp[tc==1]= (np.random.random(tc.sum())>tctp10Rate).astype(int)


    #make twice as many as needed, then just drop all the ones above 1024, and keep the ones we need
    if adcExponential:
        adcm1=(np.random.exponential(adcScale,37*12*nEvents*4)).astype(int)
        adc_or_tot=(np.random.exponential(adcScale,37*12*nEvents*4)).astype(int)
    else:
        adcm1=(np.random.normal(adcScale,adcWidth,37*12*nEvents*4)).astype(int)
        adc_or_tot=(np.random.normal(adcScale,adcWidth,37*12*nEvents*4)).astype(int)

    adcm1=adcm1[(adcm1<1024)&(adcm1>=0)][:37*12*nEvents]
    adc_or_tot=adc_or_tot[(adc_or_tot<1024)&(adc_or_tot>=0)][:37*12*nEvents]

    # find TOA for all charges above a threshold
    # 500 is just a cutoff that gives ~10% of hits with a toa for exponential chosen above
    hasTOA=((adc_or_tot>500) | tc==1)
    toa=np.zeros_like(adc_or_tot)
    toa[hasTOA]=np.random.normal(150,25,hasTOA.sum()).astype(int)

    tc=tc.reshape(nEvents,37,12)
    tp=tp.reshape(nEvents,37,12)
    adcm1=adcm1.reshape(nEvents,37,12)
    adc_or_tot=adc_or_tot.reshape(nEvents,37,12)
    toa=toa.reshape(nEvents,37,12)

    ### get common modes
    if cmExponential:
        cm=(np.random.exponential(cmScale,12*nEvents*4))
    else:
        cm=(np.random.normal(cmScale,cmWidth,12*nEvents*4))

    #random smear the cm by 10% gaussian twice to get two cm values
    cm0=cm*np.random.normal(1,.1,12*nEvents*4)
    cm1=cm*np.random.normal(1,.1,12*nEvents*4)

    cm0=cm0[(cm0<1024) & (cm0>=0)][:12*nEvents].astype(int).reshape(nEvents,12)
    cm1=cm1[(cm1<1024) & (cm1>=0)][:12*nEvents].astype(int).reshape(nEvents,12)

    #build 32 bit words
    cmData=(1<<31) + (cm0<<10) + (cm1)
    cellData=((tc<<31) + (tp<<30) + (adcm1<<20) + (adc_or_tot<<10) + toa)


    packets=np.zeros(nEvents*12*40,dtype=int).reshape(100,40,12)

    packets[:,1,:]=cmData
    packets[:,2:39,:] = cellData

    return packets

