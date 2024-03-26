
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.constants import mu_0


def forward_1D_MT(model, depths, freqs, return_G = False, return_Z = False):
    # Impedance recursive approach to compute analytically MT transfer
    # functions over a layered 1D Earth, as in:
    # Pedersen, J., Hermance, J.F. Least squares inversion of one-dimensional 
    #       magnetotelluric data: An assessment of procedures employed 
    #       by Brown University. Surv Geophys 8, 187–231 (1986).

    # adapted from https://empymod.emsig.xyz/en/stable/gallery/fdomain/magnetotelluric.html
    
    w = 2*np.pi*freqs # angular frequencies
    model_lin = 10**model # electrical resistivities in linear scale

    # Calculate impedance Z at the top of the bottom half space 
    Z = np.sqrt(1j * w * mu_0 * model_lin[-1])
    
    # The surface impedance Z is found recursively from the bottom propagating upwards
    for j in range(len(depths)-1,-1,-1):
        # calculate thickness th of layer j
        if j == 0: th = depths[j]
        else: th = depths[j] - depths[j-1]
        # calculate intrinsic impedance zo of layer j
        zo = np.sqrt(1j * w * mu_0 * model_lin[j])
        # calculate reflection coefficient R of layer j
        R = (zo - Z) / (zo + Z)
        # calculate induction parameter gamma of layer j
        gamma = np.sqrt(1j * w * mu_0 / model_lin[j])
        # update impedance Z at the top of layer j
        Z = zo * (1 - R * np.exp(-2*gamma*th)) / (1 + R * np.exp(-2*gamma*th))
        
    # instead of using the complex impedance, we use the log amplitude and phase
    # of Z (log10 apparent resistivity and phase)(Wheelock, Constable, Key; GJI 2015)
    data = z2rhophy(freqs, Z, dZ=None)
    
    if return_G:
        # Jacobian calculation using finite differences 
        M = len(model)
        N = len(data)
        G = np.zeros((N, M))
        
        # ppert = 0.01 # Perturbation (percentage)
        # for i in range(M):
        #     model_pert = model + model * ppert * np.identity(M)[:,i]
        #     dpert = forward_1D_MT(model_pert,depths, freqs)
        #     G[:,i] = (dpert - data) / (model * ppert)[i]

        apert = 0.01 # Perturbation (absolute)
        for i in range(M):
            model_pert = model + apert * np.identity(M)[:,i]
            dpert = forward_1D_MT(model_pert,depths, freqs)
            G[:,i] = (dpert - data) / apert

    if return_Z:
        return Z

    elif return_G:
        return data, G

    else:
        return data


def readEDI(file_name):
    
    """
    function that reads EDI files
    Z in linear scale; format: ohm
    
    Input:
    - complete path to the EDI file
    
    Output:
    - data (EDI pandas DF format)
    - site_id (string)
    - coord (panda DF of 7 values (['Lat_deg']['Lat_min']['Lat_sec']['Long_deg']['Long_min']['Long_sec']['Elev'])
             
    Pandas DF format:        
    # ['FREQ',
    # 'ZXXR','ZXXI','ZXX.VAR',
    # 'ZXYR','ZXYI','ZXY.VAR',
    # 'ZYXR','ZYXI','ZYX.VAR',
    # 'ZYYR','ZYYI','ZYY.VAR',
    # 'TXR.EXP','TXI.EXP','TXVAR.EXP',
    # 'TYR.EXP','TYI.EXP','TYVAR.EXP']
    
    """
    
    import numpy as np
    import re
    import pandas as pd

    coord = pd.DataFrame({'Lat_deg':0,'Lat_min':0,'Lat_sec':0,'Long_deg':0,'Long_min':0,'Long_sec':0,'Elev':0},index=[0])

    with open(file_name, 'r') as f:
        data = f.readlines()
        for i,line in enumerate(data):
            line = data[i]
            words = line.split()
            
            #READ SITE NAME
            if any("DATAID" in s for s in words):
                #words = ''.join(words)
                #print(words)
                #test = re.split('=',words)
                #site_id = test[1]
                print(file_name)
                test = re.split('/',file_name) 
                test2 = re.split('.edi',test[-1])
                print(test2)
                site_id = test2[0]
                #print(site_id)
                #site_id = re.search('Rx', words).group(1)
                #site_id = re.search('\"([^"]+)', words).group(1)

            #READ NUMBER OF FREQUENCIES
            if any("NFREQ" in s for s in words):
                words = ''.join(words)
                nfreq_str = (re.findall('\d+', words))
                nfreq = int(nfreq_str[0])

             #READ LATITUDE
            if any("REFLAT" in s for s in words):
                words = ''.join(words)
                reflat_str = (re.findall('\-?\d+', words))
                coord['Lat_deg'] = str(reflat_str[0])
                coord['Lat_min'] = str(reflat_str[1])
                coord['Lat_sec'] = str('.'.join(reflat_str[2:]))				

            #READ LONGITUDE
            if any("REFLONG" in s for s in words):
                words = ''.join(words)
                reflong_str = (re.findall('\-?\d+', words))
                coord['Long_deg'] = str(reflong_str[0])
                coord['Long_min'] = str(reflong_str[1])
                coord['Long_sec'] = str('.'.join(reflong_str[2:]))	

            #READ ELEVATION
            if any("REFELEV" in s for s in words):
                words = ''.join(words)
                refelev_str = (re.findall('\-?\d+', words))
                coord['Elev'] = str('.'.join(refelev_str[0:]))				

    #READ MT DATA
    param = ['FREQ',
             'ZXXR','ZXXI','ZXX.VAR',
             'ZXYR','ZXYI','ZXY.VAR',
             'ZYXR','ZYXI','ZYX.VAR',
             'ZYYR','ZYYI','ZYY.VAR',
             'TXR.EXP','TXI.EXP','TXVAR.EXP',
             'TYR.EXP','TYI.EXP','TYVAR.EXP',]
    
    edi_data = np.empty((nfreq, len(param)))
    
    with open(file_name, 'r') as f:
        data = f.readlines()
        for i,line in enumerate(data):
            line = data[i]
            words = line.split()
            
            for col, data_type in enumerate(param):
                aa=[]            
                if ('>%s' %data_type) in words:
                    for k in range (1,1000):                   
                        if any(">" in s for s in data[i+k].split()):
                                break
                        else:
                            a = data[i+k].split()
                            aa += a
                    edi_data[:,col] = aa
    
    
    # write to Pandas format
    edi_pd = pd.DataFrame(edi_data)
    edi_pd.columns = param

    return (edi_pd, site_id, coord)




def z2rhophy(freq,Z,dZ=None):
    
    # calcul of apparent resistivity and phases
    rho_app = abs(Z)**2 / (mu_0*2*np.pi*freq)
    #rho_app = abs(Z)**2 * (mu_0*2*np.pi*freq)**(-1)
    phase = np.degrees(np.arctan2(Z.imag,Z.real))
    
    if dZ is None:
        return rho_app, phase #, np.zeros(Z.shape[0]), np.zeros(Z.shape[0]), np.zeros(Z.shape[0])
    else:
        # calcul of errors
        drho_app = 2*rho_app*dZ / abs(Z)
        dphase = np.degrees(0.5 * (drho_app/rho_app))
        log10_drho_app = (1/np.log(10)) * (drho_app/rho_app)
        return [rho_app, phase, drho_app, dphase, log10_drho_app]



def modem2pd(df):

    params = ['FREQ',
             'ZXXR','ZXXI','ZXX.VAR',
             'ZXYR','ZXYI','ZXY.VAR',
             'ZYXR','ZYXI','ZYX.VAR',
             'ZYYR','ZYYI','ZYY.VAR',]
    
    nFreqs = len(np.unique(df['FREQ']))
    data = np.zeros((nFreqs, len(params)))
    dfZ = pd.DataFrame(data)
    dfZ.columns = params
    
    dfZ['FREQ'] = pd.unique(df['FREQ'])
    dfZ['ZXXR'] = (df['REAL'][df['COMP']=='ZXX']).values
    dfZ['ZXXI'] = (df['IMAG'][df['COMP']=='ZXX']).values
    dfZ['ZXX.VAR'] = (df['STD'][df['COMP']=='ZXX']).values
    dfZ['ZXYR'] = (df['REAL'][df['COMP']=='ZXY']).values
    dfZ['ZXYI'] = (df['IMAG'][df['COMP']=='ZXY']).values
    dfZ['ZXY.VAR'] = (df['STD'][df['COMP']=='ZXY']).values
    dfZ['ZYXR'] = (df['REAL'][df['COMP']=='ZYX']).values
    dfZ['ZYXI'] = (df['IMAG'][df['COMP']=='ZYX']).values
    dfZ['ZYX.VAR'] = (df['STD'][df['COMP']=='ZYX']).values
    dfZ['ZYYR'] = (df['REAL'][df['COMP']=='ZYY']).values
    dfZ['ZYYI'] = (df['IMAG'][df['COMP']=='ZYY']).values
    dfZ['ZYY.VAR'] = (df['STD'][df['COMP']=='ZYY']).values

    return dfZ


def Zdet(Z):
    
    """
    Compute the determinant of Z
    
    Zdet = Zxx*Zyy - Zxy*Zyx = Zdet_1 - Zdet_2
    """
    nF = len(Z)
    # Calculate determinant 
    mat = np.zeros((2,2,nF), complex)
    mat[0,0,:] = Z['ZXXR'].values + (Z['ZXXI'].values * 1j)
    mat[0,1,:] = Z['ZXYR'].values + (Z['ZXYI'].values * 1j)
    mat[1,0,:] = Z['ZYXR'].values + (Z['ZYXI'].values * 1j)
    mat[1,1,:] = Z['ZYYR'].values + (Z['ZYYI'].values * 1j)
    
    ZdetR = np.zeros((nF))
    ZdetI = np.zeros((nF))
    sd = np.zeros((nF))
    lnSd = np.zeros((nF))
    
    for freq_det in range(len(Z)):
        det = np.linalg.det(mat[:,:,freq_det])**0.5
        ZdetR[freq_det] = det.real
        ZdetI[freq_det] = det.imag
    
    
    # Calculate determinant std dev.
    # (log transform of) x + dx --> log(x) + (log(1+dx/x) - log(1-dx/x))/(2*sqrt(2))
    cent = (ZdetR**2 + ZdetI**2)**0.5
    sd = (Z['ZXX.VAR'] + Z['ZXY.VAR'] + Z['ZYX.VAR'] + Z['ZYY.VAR'])**0.5
    lnSd = (np.log(1+sd/cent) - np.log(1-sd/cent))/(2*np.sqrt(2))
    
    return ZdetR ,ZdetI, sd, lnSd






def add_noise(Z, percentage = 5, seed = 1234):
    # The standard deviations are taken as a percentage of the amplitude |Z|.
    # We assume a circularly symmetric Gaussian distribution in the complex 
    # plane for Z, with a common standard deviation Zerr for the real and
    # imaginary parts.
    
    np.random.seed(seed)
    Zerr = np.zeros_like(Z, dtype=float)
    for f in range(Z.shape[0]):
        Zerr[f] = 0.01*percentage*(Z[f].real**2+Z[f].imag**2)**0.5
        Zr = Z[f].real + np.random.normal(0.0, 0.01*percentage*abs(Z[f]))
        Zi = Z[f].imag + np.random.normal(0.0, 0.01*percentage*abs(Z[f]))
        Z[f] = Zr + 1j*Zi
    return Z, Zerr



    
def plot_1D_model(model, depths, color='g', label='1D resistivity model'):
    plt.figure(10,figsize=(5,8))

    plt.loglog(model,depths, color + '-', label=label)
    # plt.ylim(0, 0.2)
    # plt.xticks([])
    
    plt.ylim(10,100000)
   
    plt.legend(loc=3,framealpha = 1, edgecolor='k')
    plt.ylabel('Depth (m)')
    plt.xlabel ('Resistivity (ohm.m)')
    plt.grid(linestyle=':')
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', top=True, bottom=False)
    



    
