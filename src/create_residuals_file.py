# -*- coding: utf-8 -*-
"""
Created on Fri Dec 07 13:55:29 2018

@author: sei029
"""

# script that extract a conductivity column beneath a list of MT site on a 3D grid 
# and perform a 1D fwd

# save the residuals and data as a file for input into mtul.jl

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import cmath
from scipy.constants import mu_0


import mt as mt
import modem3d as modem

# x is east
# y is north

#1 read modem 3D fwd file and 3D model file
model_name = 'modelval02'
model = './model_3D/%s.mod'%model_name
fwd_file = './model_3D/%s.dat'%model_name

m1, dx,dy,dz = modem.read_model(model)#, inv_result = False)
dataMT, list_MTsites, list_freqs, list_coordinates, list_comps = modem.read_dat(fwd_file,lenght_header=8)
xyMT= np.array(np.c_[list_coordinates[:,4],list_coordinates[:,3]],dtype=float) # y(north) then x(east)
model_3D =  m1

#2 extract 1D column beneath each site

#2.1 extract cell centers coordinates (x,y)
x_length = np.sum(dx)
x_centers = np.zeros((len(dx)))
x_centers[0] = -0.5*x_length + 0.5*dx[0]
x_edges = np.zeros((len(dx)+1))
x_edges[0] = x_centers[0] - (0.5*dx[0])
for i in range(1, len(dx)):
    x_centers[i] = x_centers[i-1] + (0.5*dx[i-1] + 0.5*dx[i])
    x_edges[i] = x_centers[i] - (0.5*dx[i])
x_edges[-1] = x_centers[-1] + 0.5*dx[-1]

y_length = np.sum(dy)
y_centers = np.zeros((len(dy)))
y_centers[0] = -0.5*y_length + 0.5*dy[0]
y_edges = np.zeros((len(dy)+1))
y_edges[0] = y_centers[0] - (0.5*dy[0])
for j in range(1, len(dy)):
    y_centers[j] = y_centers[j-1] + (0.5*dy[j-1] + 0.5*dy[j])
    y_edges[j] = y_centers[j] - (0.5*dy[j])
y_edges[-1] = y_centers[-1] + 0.5*dy[-1]

xc, yc = np.meshgrid(x_centers, y_centers[::-1])#, indexing='ij')
xc_idx, yc_idx = np.meshgrid(np.linspace(0,len(dx)-1,len(dx),dtype=int) , np.linspace(0,len(dy)-1,len(dy),dtype=int))#, indexing='ij')
coord_cells = np.c_[xc.ravel(), yc.ravel(),xc_idx.ravel(), yc_idx.ravel()]


# check plot
plt.figure(j,figsize=(12,12))
plt.pcolormesh(x_edges,
                y_edges[::-1],
                (np.log10(model_3D[:,:,1])).T,
                edgecolors = 'k',cmap='jet_r')
    
plt.plot(xyMT[:,0],xyMT[:,1],'k+',ms=5)

for i in range(len(list_MTsites)):
    plt.text(xyMT[i,0],xyMT[i,1],s=list_MTsites[i],ha="center")
    
# plt.plot(coord_cells[:,0],coord_cells[:,1],'r.')
# for i in range(len(coord_cells)):
#     plt.text(coord_cells[i,0],coord_cells[i,1],s='(%d,%d)'%(coord_cells[i,2],coord_cells[i,3]),ha="center")

plt.axis('equal')
plt.xlim(xyMT[:,0].min()-1000,xyMT[:,0].max()+1000)
plt.ylim(xyMT[:,1].min()-1000,xyMT[:,1].max()+1000)



#2.2 use kdree to find closest cell center to each MT site
from scipy.spatial import KDTree
tree = KDTree(coord_cells[:,0:2])
dd, ii = tree.query(xyMT, k=1)
idx_1D = np.c_[coord_cells[ii,2],coord_cells[ii,3]]


#3 compute 1D fwd beneath each MT site
depths = np.cumsum(dz[:-1])
depths2 = np.insert(depths, 0, 0)
resp_1D = []
models = []
sorted_list_freqs = np.sort(list_freqs)

for site in range(len(list_MTsites)):
    model = np.log10(model_3D[int(idx_1D[site,0]),int(idx_1D[site,1]),:])
    models.append(model)
    # model = np.log10(model_3D[int(idx_1D[site,0]),int(idx_1D[site,1]),:])
    
    #Z = mt.forward_1D_MT(model, depths, sorted_list_freqs, return_Z = True)
    Z = mt.forward_1D_MT(model, depths, sorted_list_freqs,  return_G = False,return_Z = False)
    resp_1D.append([list_MTsites[site],Z])
    #mt.plot_1D_model(model, depths2)
    #plt.savefig('./plots/1D_%s_%s.jpg'%(model_name,list_MTsites[site]),dpi=300)
    #plt.close()
    
#df=pd.DataFrame(models)
#csv_file_path = '1D_models.csv'
#f.to_csv(csv_file_path, sep=',', index=False)

##############################
#4 compute phase tensor of the 3D response for each MT site
ph_tensor = []
fwd_3Ddata = []
for site in range(len(list_MTsites)):
    dfZ1 = mt.modem2pd(dataMT[site])
    dfZ = dfZ1.sort_values(by='FREQ',ascending=False)
    # dfZ = addNoise
    ph_tens, ph_params = mt.phaseTensor(dfZ)
    ph_tensor.append([list_MTsites[site], ph_params[:,[4,5,6]]])
    fwd_3Ddata.append([list_MTsites[site], dfZ])
###############

#5 write to file (and plot)

plot_data = False
#data_mtul = '%s.csv'%(model_name) 

#file = open(data_mtul,'w')
#file.write('site,freq(Hz),'
#     'Zr,Zi\n')

data_rho = np.zeros((420,24))
data_phi = np.zeros((420,24))

for site in range(len(list_MTsites)):
# for site in range(1):
    dfZ = fwd_3Ddata[site][1]
    resp_1D_site = resp_1D[site][1]
    site_name = fwd_3Ddata[site][0]
    assert fwd_3Ddata[site][0] == resp_1D[site][0]
    
    #receiver_file = './training_data/%s_%s.csv'%(model_name,site_name) 

    #file2 = open(receiver_file,'w')
    #file2.write('site,freq(Hz),'
    #     'rho,phi\n')
    
    
    X_utm = list_coordinates[site,3]
    Y_utm = list_coordinates[site,4]
    
    beta = ph_tensor[site][1][:,0]
    ellip = ph_tensor[site][1][:,1]
    azimuth = ph_tensor[site][1][:,2]
    
    #5 compute residuals
    Zdet_3D_real, Zdet_3D_imag, _, _ = mt.Zdet(dfZ)
    #resR = abs(np.log10(resp_1D_site.real) - np.log10(Zdet_3D_real/(10000.0/(4*np.pi))))
    #resI = abs(np.log10(resp_1D_site.imag) - np.log10(Zdet_3D_imag/(10000.0/(4*np.pi))))
    
    #sumR = np.sum(resR+resI)
    
  


    if plot_data:
        if not os.path.isdir(r'./plots'):
            os.makedirs(r'./plots')
        #mt.plot_rho_phi(dfZ,resp_1D_site,Zdet_3D_real, Zdet_3D_imag,site_name,beta,ellip,resR,resI)
        plt.savefig('./plots/%s_%s.jpg'%(model_name,site_name),dpi=300)
        plt.close()
        
        


    
    for freq in range(0, len(dfZ)):
        
       #rho = np.log10(((resp_1D_site[freq].real**2+resp_1D_site[freq].imag**2))*1/(mu_0*2*np.pi*(dfZ['FREQ'].values[freq])))
       #phi = np.degrees(np.arctan2(resp_1D_site[freq].imag,resp_1D_site[freq].real))
       
       rho = resp_1D_site[0][freq]
       phi = resp_1D_site[1][freq]
       
       #np.hstack(data_rho,rho)
       data_rho[site,freq]=np.log10(rho)
       data_phi[site,freq]=phi
       
       #file2.write('%s,%4.4f,\
        #           %2.5f,%2.5f\n'\
         #          %(site_name, dfZ['FREQ'].values[freq], 
                     #dfZ['ZXXR'].values[freq]/(10000.0/(4*np.pi)), dfZ['ZXXI'].values[freq]/(10000.0/(4*np.pi)),
                     #dfZ['ZXYR'].values[freq]/(10000.0/(4*np.pi)), dfZ['ZXYI'].values[freq]/(10000.0/(4*np.pi)),
                     #dfZ['ZYXR'].values[freq]/(10000.0/(4*np.pi)), dfZ['ZYXI'].values[freq]/(10000.0/(4*np.pi)),
                     #dfZ['ZYYR'].values[freq]/(10000.0/(4*np.pi)), dfZ['ZYYI'].values[freq]/(10000.0/(4*np.pi)),
                     #beta[freq], ellip[freq],azimuth[freq],
          #           np.log10(rho),phi))

    #file2.close()
    
    #data_rho.append(data_rho)
    #data_phi.append(data_phi)

df=pd.DataFrame(models)
df2=pd.DataFrame(data_rho)
df3=pd.DataFrame(data_phi)


# Replace the index with custom names
custom_names = ['%s_%s'%(model_name,site) for site in range(len(list_MTsites))]
df.index = custom_names
df2.index = custom_names
df3.index = custom_names

csv_file_path = './model_1D/1D_models_%s.csv' %(model_name)
csv_file_path2 = './training_data/data_app_rho_%s.csv'%(model_name)
csv_file_path3 = './training_data/data_phi_%s.csv'%(model_name)

df.to_csv(csv_file_path, sep=',', index=True)
df2.to_csv(csv_file_path2, sep=',', index=True)
df3.to_csv(csv_file_path3, sep=',', index=True)
