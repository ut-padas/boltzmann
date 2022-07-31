"""
@package To build efficient interpolation methods to deal with the experimental cross section data. 
"""

import numpy as np
import typing as tp
# this is the read lxCat data file.
import lxcat_data_parser as ldp
from scipy import interpolate

def lxcat_cross_section_to_numpy(file : str, column_fields : tp.List[str] )->list:
    data    = ldp.CrossSectionSet(file)
    np_data = list()
    for f in column_fields:
        np_data.append(data.cross_sections[0].data[f].to_numpy())
    
    return np_data

#import matplotlib.pyplot as plt
# np_data = lxcat_cross_section_to_numpy("lxcat_data/e_Ar_elastic.txt",["energy","cross section"])
# #print(np_data[0])
# #print(np_data[1])
# x = np_data[0]
# y = np_data[1]

# plt.plot(x,y)
# plt.xscale('log')

# #xnew = np.linspace(np.min(x), np.max(x), 1000000) 
# print(x.shape)
# xnew=np.zeros(np.size(x)*2)
# print(xnew.shape)
# for i in range(np.size(x)):
#     xnew[2*i] = x[i] 
#     if i < (np.size(x)-1):
#         print("i: %d",i)
#         xnew[2*i + 1] = (x[i] + x[i+1])/2

# print(xnew)

# spl = interpolate.interp1d(x,y,kind='linear')
# power_smooth = spl(xnew[:-1])

# plt.plot(xnew[:-1], power_smooth)
# plt.show()





