#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of how to load previously saved parameters of a trained model and evaluate
performance of model on new data
"""

import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_function import *
from format_data import plot_scatter
import h5py


#load trained model
model = ModelExtraInput() #create instance of model
model.load_state_dict(torch.load('model_multimass_1.pt')) #load previously saved parameters

title = ['1 Mev','5 MeV','10 MeV','50 MeV','100 MeV','500 MeV','1000 MeV','Background',]
datasets = ['signal/ecal/m1/xyz_64','signal/ecal/m5/xyz_64','signal/ecal/m10/xyz_64',
    'signal/ecal/m50/xyz_64','signal/ecal/m100/xyz_64','signal/ecal/m500/xyz_64', 'signal/ecal/m1000/xyz_64', 'background/ecal/data1/xyz_64']

#original list of hits (not reconstructed images)
#m_ori_file_tab = ['data/ecal/signal/ecal_sign_10_test.npy','data/ecal/signal/ecal_sign_10.npy']
mass_tab = [1,5,10,50,100,500,1000,0]

net_name = 'model_multimass_1'

all_transforms = transforms.Compose([transforms.ToTensor()])

for j in range(7): #evaluate trained network on data
    
    with h5py.File('dataset.hdf5', 'a') as f:
        n_test = len(f[datasets[j]][:,0,0,0])
    
    nam = title[j] #description to ass as title to figures

    all_transforms = transforms.Compose([transforms.ToTensor()]) #extra transf. to apply to images

    event_test = my_Dataset_test_single(n_test,datasets[j],mass_tab[j],transform=all_transforms) #create test dataset

    test_loader = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=False) #create iterator

    outs,labels,im_tab,m_tab = test_any_model(model,test_loader,nam,batch_size,dim,depth=3) #evaluate model

    # append group with new dataset with results
    with h5py.File('dataset.hdf5', 'a') as f:
        #append file with array of images
        new_data = f.create_dataset(datasets[j]+'/'+net_name,data=outs)
        new_data.attrs['date'] = time.time()