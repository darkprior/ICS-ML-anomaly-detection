# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:37:02 2023

@author: Denis
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import pickle

listok=[]
hlavickabin=[]
test=[]
abin,bbin,cbin,dbin,ebin,fbin,gbin,hbin,ibin,jbin,kbin,lbin,mbin,nbin,obin,pbin=([] for i in range(16))
# with open('plcclanok/nove/zmenaprvotny.txt', 'r') as fp:
with open('plcclanok/nove/zmena5.txt', 'r') as fp:
    hex_list = ["{:02x}".format(ord(c)) for c in fp.read()]
# with open('plcclanok/nove/zmenaprvotny.txt') as f,open('okbin.txt', 'w') as f_out:
with open('plcclanok/nove/zmena5.txt') as f,open('okbin.txt', 'w') as f_out:
    for line in f:
        line = line.strip()
        listok.append(line)

scale = 16 ## equals to hexadecimal

num_of_bits = 8
def hexnadec(hex):
    """Convert a hexadecimal string to a decimal number"""
    result_dec = int(hex, 16)
    return result_dec

for i in range(0,len(listok)):

    # hlavickabin.append(bin(int(listok[i][0:4], scale))[2:].zfill(num_of_bits))
    abin.append(hexnadec(listok[i][7:9]))
    bbin.append(hexnadec(listok[i][10:12]))
    cbin.append(hexnadec(listok[i][13:15]))
    dbin.append(hexnadec(listok[i][16:18]))
    ebin.append(hexnadec(listok[i][19:21]))
    fbin.append(hexnadec(listok[i][22:24]))
    gbin.append(hexnadec(listok[i][25:27]))
    hbin.append(hexnadec(listok[i][28:30]))
    ibin.append(hexnadec(listok[i][31:33]))
    jbin.append(hexnadec(listok[i][34:36]))
    kbin.append(hexnadec(listok[i][37:39]))
    lbin.append(hexnadec(listok[i][40:42]))
    mbin.append(hexnadec(listok[i][43:45]))
    nbin.append(hexnadec(listok[i][46:48]))
    obin.append(hexnadec(listok[i][49:51]))
    pbin.append(hexnadec(listok[i][52:54]))
    # test.append(abin[i]+bbin[i]+cbin[i]+dbin[i]+ebin[i]+fbin[i]+gbin[i]+hbin[i]+ibin[i]+jbin[i]+kbin[i]+lbin[i]+mbin[i]+nbin[i]+obin[i]+pbin[i])

dokopy=np.column_stack((abin,cbin,dbin,ebin,fbin,gbin,hbin,ibin,jbin,kbin,lbin,mbin,nbin,obin,pbin))
# dokopy=np.concatenate((abin,bbin,cbin,dbin,ebin,fbin,gbin,hbin,ibin,jbin,kbin,lbin,mbin,nbin,obin,pbin))
#%%
# test=(abin[0]+bbin[0]+cbin[0]+dbin[0]+ebin[0]+fbin[0]+gbin[0]+hbin[0]+ibin[0]+jbin[0]+kbin[0]+lbin[0]+mbin[0]+nbin[0]+obin[0]+pbin[0]).format('16*8d')
# test2=(abin[1]+bbin[1]+cbin[1]+dbin[1]+ebin[1]+fbin[1]+gbin[1]+hbin[1]+ibin[1]+jbin[1]+kbin[1]+lbin[1]+mbin[1]+nbin[1]+obin[1]+pbin[1]).format('16*8d')

# testt=np.vstack((test,test2)).format('16*8d')

# testt=np.asarray(test)



np.savetxt('plcclanok/train/zmena_dokopy15/zmena5.dat',dokopy,fmt='%i')

# testdata=np.loadtxt('plcclanok/nove/ok1.dat')
#%%
# nacitanie dat

# data1=np.loadtxt('plcclanok/train/ok/ok2.dat',dtype=(int))
# data2=np.loadtxt('plcclanok/train/ok/ok1.dat',dtype=(int))
# dataa=np.row_stack((data1,data2))
# nacitane=[]
# with open('plcclanok/train/ok/ok1.dat') as my_file:
#     testsite_array = my_file.readlines()


# for i in range(0,len(testsite_array)):
#     nacitane.append(int(testsite_array[i],0))
    
    
    
