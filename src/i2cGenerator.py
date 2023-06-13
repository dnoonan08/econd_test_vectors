import numpy as np

def getParam_str_to_int(param):
    if isinstance(param,str):
        if set(param)=={'0','1'}:
            return int(param,2)
        else:
            return int(param,16)
    else:
        return param

def randomize_eRx_eTx_settings(yamlName=None,
                               minERx=3,
                               maxERx=12,
                               minETx=1,
                               maxETx=6,
                               matchThresh=None,
                               straightCMRoute=False
                              ):
    i2c={}
    #randomize the nubmer of eRx, requiring at least 3
    i2c['ERx_active']=np.random.randint(0,2,12).astype(bool)
    while (i2c['ERx_active'].sum()<minERx) or (i2c['ERx_active'].sum()>maxERx):
        i2c['ERx_active']=np.random.randint(0,2,12).astype(bool)

    #randomize the nubmer of eTx, requiring at least 1
    i2c['ETx_active']=np.random.randint(0,2,6).astype(bool)
    while (i2c['ETx_active'].sum()<minETx) or (i2c['ETx_active'].sum()>maxETx):
        i2c['ETx_active']=np.random.randint(0,2,6).astype(bool)

    if matchThresh is None:
        #randomize match and vRecon thresholds, must be greater than half the active eLinks
        i2c['Match_thresh']=np.random.randint(i2c['ERx_active'].sum()/2,i2c['ERx_active'].sum())
        i2c['VReconstruct_thresh']=np.random.randint(i2c['ERx_active'].sum()/2,i2c['ERx_active'].sum())
    else:
        i2c['Match_thresh']=int(matchThresh)
        i2c['VReconstruct_thresh']=int(matchThresh)


    if straightCMRoute:
        #route common modes straight through
        i2c['CM_eRX_Route']=np.arange(12)
        i2c['CM_Selections']=np.array([0,0,1,1,2,2,3,3,4,4,5,5])
    else:
        #randomly route the eRx to the common modes
        i2c['CM_eRX_Route']=np.random.choice(np.argwhere(i2c['ERx_active']).flatten(),12)
        #randomize selection of which CM average to use
        i2c['CM_Selections']=np.random.randint(0,8,12)
    #randomize the user defined CM
    i2c['CM_UserDef']=np.random.randint(0,1024,12)

    if not yamlName is None:
        convertI2CtoYAML(i2c,yamlName)

    return i2c

def randomize_formatter_patterns(yamlName=None,
                                 headerMarker=None,
                                 idlePattern=None
                                ):
    i2c={}
    if headerMarker is None:
        #randomize each of the bits independently
        i2c['HeaderMarker']=(np.random.randint(0,2,5) << np.arange(5)).sum()
    else:
        i2c['HeaderMarker']=getParam_str_to_int(headerMarker)

    if idlePattern is None:
        i2c['IdlePattern']=(np.random.randint(0,2,24) << np.arange(24)).sum()
    else:
        i2c['IdlePattern']=getParam_str_to_int(idlePattern)

    if not yamlName is None:
        convertI2CtoYAML(i2c,yamlName)

    return i2c

def randomize_ELinkProcessor_settings(yamlName=None,
                                      eboRecoMode=0,
                                      passThru=0,
                                      simpleMode=1,
                                      zs_ce=None,
                                      zs_ce_range=[0,1024],
                                      zs_c=None,
                                      zs_c_range=[0,256],
                                      zs_kappa=None,
                                      zs_kappa_range=[0,64],
                                      zs_lambda=None,
                                      zs_lambda_range=[0,128],
                                      zs_mask=None,
                                      zs_mask_rate=0.1,
                                      zs_pass=None,
                                      zs_pass_rate=0.1,
                                     ):

    i2c={}
    i2c['EBO_ReconMode'] = eboRecoMode
    i2c['PassThruMode'] = passThru
    i2c['SimpleMode'] = simpleMode

    if zs_ce is None:
        i2c['ZS_ce']=np.random.randint(zs_ce_range[0],zs_ce_range[1])
    else:
        i2c['ZS_ce']=getParam_str_to_int(zs_ce)

    if zs_c is None:
        i2c['ZS_c']=np.random.randint(zs_c_range[0],zs_c_range[1],12*37).reshape(12,37)
    else:
        if isinstance(zs_c,(list,np.ndarray)):
            i2c['ZS_c']=np.array(zs_c).reshape(12,37)
        elif isinstance(zs_c,(int, str)):
            i2c['ZS_c']=np.array([getParam_str_to_int(zs_c)]*12*37).reshape(12,37)

    if zs_kappa is None:
        i2c['ZS_kappa']=np.random.randint(zs_kappa_range[0],zs_kappa_range[1],12*37).reshape(12,37)
    else:
        if isinstance(zs_kappa,(list,np.ndarray)):
            i2c['ZS_kappa']=np.array(zs_kappa).reshape(12,37)
        elif isinstance(zs_kappa,(int, str)):
            i2c['ZS_kappa']=np.array([getParam_str_to_int(zs_kappa)]*12*37).reshape(12,37)

    if zs_lambda is None:
        i2c['ZS_lambda']=np.random.randint(zs_lambda_range[0],zs_lambda_range[1],12*37).reshape(12,37)
    else:
        if isinstance(zs_lambda,(list,np.ndarray)):
            i2c['ZS_lambda']=np.array(zs_lambda).reshape(12,37)
        elif isinstance(zs_lambda,(int, str)):
            i2c['ZS_lambda']=np.array([getParam_str_to_int(zs_lambda)]*12*37).reshape(12,37)

    if zs_mask is None:
        i2c['ZS_mask']=(np.random.rand(37*12)<zs_mask_rate).astype(int).reshape(12,37)
    else:
        if isinstance(zs_mask,(list,np.ndarray)):
            i2c['ZS_mask']=np.array(zs_mask).reshape(12,37)
        elif isinstance(zs_mask,(int, str)):
            i2c['ZS_mask']=np.array([getParam_str_to_int(zs_mask)]*12*37).reshape(12,37)

    if zs_pass is None:
        i2c['ZS_pass']=(np.random.rand(37*12)<zs_pass_rate).astype(int).reshape(12,37)
    else:
        if isinstance(zs_pass,(list,np.ndarray)):
            i2c['ZS_pass']=np.array(zs_pass).reshape(12,37)
        elif isinstance(zs_pass,(int, str)):
            i2c['ZS_pass']=np.array([getParam_str_to_int(zs_pass)]*12*37).reshape(12,37)

    return i2c
