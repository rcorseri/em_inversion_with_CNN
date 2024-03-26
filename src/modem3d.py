# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:31:37 2022

@author: sei029
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import ListedColormap



def read_dat(data_file,lenght_header = 12):

    df = pd.read_table(data_file, 
                       skiprows=lenght_header, 
                       delim_whitespace=True, 
                       header = None,
                       dtype=str,) 
                       #error_bad_lines='skip')
                       #error_bad_lines=False)
    df = df.dropna()
    df = df[~df.isin(['Period(s)']).any(axis=1)]
    
    df.columns = ['FREQ','ID','LAT','LONG','X','Y','Z','COMP','REAL','IMAG','STD']
    df = df.astype({"FREQ": float,
                    "ID": str,
                    "LAT": float,
                    "LONG": float,
                    "X": float,
                    "Y": float,
                    "Z": float,
                    "COMP": str,
                    "REAL": float,
                    "IMAG": float,
                    "STD": float})


    list_MTsites  = df.ID.unique()
    list_comps    = df.COMP.unique() 
    list_freqs    = 1/df.FREQ.unique()
    
    list_coordinates = []
    dataMT = []
    for site in list_MTsites:
        df_site = df[df['ID']==site]
        df_site = df_site.reset_index(drop=True)
        list_coordinates.append([df_site['ID'][0],
                          df_site['LAT'][0],df_site['LONG'][0],
                          df_site['X'][0],df_site['Y'][0],df_site['Z'][0]])
        df_site = df_site.drop(columns=['LAT','LONG','X','Y','Z'])
        df_site['FREQ'] = 1/df_site['FREQ'].values
        dataMT.append(df_site)
    list_coordinates = np.array(list_coordinates)
    
    return dataMT, list_MTsites, list_freqs, list_coordinates, list_comps
    



def read_model(model):#, inv_result = True):
 
    '''
    model m is read that way:
        rows = from west to east (X)
        columns = from north to south (Y)
        
    '''

    data = open(model,'r')

    header = next(data)
    raw_data = data.read().strip().split()
        
    #read mesh size
    ny = int(raw_data[0])  #NS
    nx = int(raw_data[1])  #EW
    nz = int(raw_data[2])  #Z
    
    #read cell sizes in 3 directions
    dx = np.zeros(nx)
    dy = np.zeros(ny)
    dz = np.zeros(nz)
    
    for yy in range(ny):
      dy[yy] = raw_data[yy + 5]
    for xx in range(nx):
      dx[xx] = raw_data[xx+5+ny]
    for zz in range(nz):
      dz[zz] = raw_data[zz+5+nx+ny]
    
    #read model values
    m = np.zeros((nx, ny, nz))
    
    #crop header values and last winglink mesh info (20 "words")
    raw_data_rho = raw_data[5+nx+ny+nz:]
    
    ind = 0
    for zz in range(nz):
        for xx in range(nx):
            for yy in range(ny):
                m[xx,yy,zz] = raw_data_rho[ind]
                ind += 1 
                
    m = np.exp(m)      # convert to linear resistivity values
    dy = dy[::-1]

    return m, dx,dy,dz

