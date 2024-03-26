# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:59:37 2024

@author: romain
"""


### Test with field data
df4,b,c = mt.readEDI('./tests/field_data/BS1305_H5_14Rx036a.edi')
ZR,ZI,dd,ff=mt.Zdet(df4)
test2 = df4.to_numpy()
freq = test2[:,0]
rho = ((ZR**2+ZI**2)*0.2/(freq))
#rho_app = abs(Z)**2 * (mu_0*2*np.pi*freq)**(-1)
phase = np.degrees(np.arctan2(ZI,ZR))


#test4=np.concatenate([rho[ np.newaxis,:,np.newaxis], phase[np.newaxis,:,np.newaxis]], axis=2)
test4 =  rho[np.newaxis,:,np.newaxis]
test5=test4.transpose((0,2,1))


field_tensor_rho = torch.tensor(
 test5,
    dtype=torch.float32
)

field_tensor_rho = field_tensor_rho.permute(0, 2, 1)


mean2 = field_tensor_rho.mean(dim=(0, 1), keepdim=True)
std2 = field_tensor_rho.std(dim=(0, 1), keepdim=True)
# Create a normalization transform
normalize = transforms.Normalize(mean=mean2, std=std2)

# Apply normalization to training and test sets
field_tensor_rho = normalize(field_tensor_rho)



test8 =  phase[np.newaxis,:,np.newaxis]
test9=test8.transpose((0,2,1))


field_tensor_phase= torch.tensor(
 test5,
    dtype=torch.float32
)

field_tensor_phase = field_tensor_phase.permute(0, 2, 1)


###Field data plot model

mean3 = field_tensor_phase.mean(dim=(0, 1), keepdim=True)
std3 = field_tensor_phase.std(dim=(0, 1), keepdim=True)
# Create a normalization transform
normalize = transforms.Normalize(mean=mean3, std=std3)

# Apply normalization to training and test sets
field_tensor_phase = normalize(field_tensor_phase)
test7 = model(field_tensor_rho[:, :, :],field_tensor_phase[:, :, :])



plt.semilogx(depths, test7.detach().numpy(), 'gd')  # , label='Predicted Data')
plt.legend()
plt.show()
