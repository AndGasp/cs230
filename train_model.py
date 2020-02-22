#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example code of how to create dataset, train model and evaluate it.
"""

import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_functions import *
from format_data import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Hyperparameters
num_epochs = 5
batch_size = 256
learning_rate = 0.0005


name = 'simple 4 layer CNN with mass input, RBG-XYZ projections \n64x64, normalized per event' #description to add as title to figures

n_train = 1000000 #number of background and signal events (equal for now) to train on
n_test = 50000 #number of background and signal events (equal for now) to test


#Compose extra transformation to apply to data, minimally need to convert image to tensor
all_transforms = transforms.Compose([transforms.ToTensor()])

#Initiate training and test datasets
event_train = my_Dataset_train(n_train,['signal/ecal/m1/xyz_64','signal/ecal/m10/xyz_64','signal/ecal/m100/xyz_64','signal/ecal/m1000/xyz_64'],'background/ecal/m1000/xyz_64',[1,10,100,1000],transform=all_transforms) #create training dataset
event_test = my_Dataset_test(n_test,['signal/ecal/m1/xyz_64','signal/ecal/m10/xyz_64','signal/ecal/m100/xyz_64','signal/ecal/m1000/xyz_64'],'background/ecal/m1000/xyz_64',[1,10,100,1000],transform=all_transforms) #create test dataset

#Create event iterators
train_loader_1 = DataLoader(dataset=event_train, batch_size=batch_size, shuffle=True) #create iterator for training and testinsg
test_loader_1 = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True)


model = ModelExtraInput() #create instance of model
#If want to train without using mass information, use ConvNet()

#run training and testing
outs , labels , images , masses = train_test_cnn_multiple_m(model,train_loader_1,test_loader_1,num_epochs,learning_rate,batch_size,name,dim,depth=3)
#if using ConvNet(), need to use train_test_cnn_single_m , same arguments


#Create histogram and ROC curves for he output on this testing dataset
area, thresh = roc_curve(labels,outs[:,1],num_epochs,learning_rate,batch_size,name)

#computes fraction of correctly classified events, fraction of signal passing cuts
#and fraction of events passing cuts that are actually signal
accur,accept,sign_frac,back_sup = accuracy(labels,outs[:,1],thresh)


print('optimal threshold = {:.4f}, {:.4f} misclassified events ({:.4f}%), accuracy of {:.4f}, Signal acceptance of {:.4f}%, background sup. of {:.4f}'.format(thresh,n_false,accur,accept,sign_frac,back_sup))

#save parameters of trained model
torch.save(model.state_dict(), 'trained_models/model_multimass.pt')
