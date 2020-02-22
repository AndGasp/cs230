#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset definitions, model definitions, function to train, test and evaluate models, 
to produce ROC curves and compute accuracy of models.
"""
import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from format_data import plot_scatter
import h5py


class my_Dataset_train(Dataset):
    """
    Defined for a training dataset of n signal events and n background events 
    to train on all masses at once (from file with 1.1 million background events and signal files with 300 000 events each)

    Used to define training datasets

    Can also accept defined torchvision transforms to apply the data before training (minimally need to
    convert data to pytorch tensor, normalisation)
    
    len defined to get number of events

    getitem defined to use dataset for iterators over events (Dataloader)
    """

    def __init__(self,n,signal_dataset_list,backround_dataset,m_list,transform=None):
        #initialize dataset
        self.n = n
        self.hdf5_file = h5py.File('dataset.hdf5', 'r')
        self.signal_groups = signal_dataset_list 
        self.background_group = background_dataset
        self.n_m = len(signal_group_list)
        self.n_per_m = self.n/len(signal_group_list) #number of event per signal mass point
        self.transform = transform
        self.m_list = m_list

        #get dimensions of image
        one_image = self.hdf5_file[background_dataset][0,:,:,:]
        self.dim = one_image[:,0,0]
        self.depth = one_image[0,0,:]


    def __len__(self):
        return 2*self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_batch = len(idx)
        events = np.zeros(n_batch,self.dim,self.dim,self.depth)
        masses = np.zeros(n_batch)
        labels = np.zeros(n_batch)

        j = 0

        for i in idx:

            if i<self.n:
                #signal event
                group = self.signal_groups[i // self.n_per_m]
                image_index = i % self.n_per_m
                labels[j] = 1
                masses[j] = self.m_list[i // self.n_per_m]

            if i>=self.n:
                group = self.background_group
                image_index = i-self.n
                labels[j] = 0

                m_randint = np.random.randint(self.n_m)
                masses[j] = self.m_list[m_randint] #random mass for background 

            dataset = self.hdf5_file[group] #pointer to dataset in disk
            events[j,:,:,:] = dataset[image_index,:,:,:] #locally save desired image

            j+=1

        #convert to tensors
        labels = torch.from_numpy(labels)
        masses = torch.from_numpy(masses).double()

        if self.transform:

            events = self.transform(events) #minimallt convert to tensors

        return (events,labels,masses)


class my_Dataset_test(Dataset):
    """
    Defined for a test dataset of 40 000 signal events and 40 000 background events SPECIFICALLY, 
    to train on all masses at once (from file with 1.1 million background events and signal files with 300 000 events each)

    Used to define testing datasets

    Can also accept defined torchvision transforms to apply the data before training (minimally need to
    convert data to pytorch tensor, normalisation)
    
    len defined to get number of events

    getitem defined to use dataset for iterators over events (Dataloader)
    """

    def __init__(self,n,signal_dataset_list,backround_dataset,m_list,transform=None):
        #initialize dataset
        self.n = n
        self.hdf5_file = h5py.File('dataset.hdf5', 'r')
        self.signal_groups = signal_dataset_list 
        self.background_group = background_dataset
        self.n_m = len(signal_group_list)
        self.n_per_m = self.n/len(signal_group_list) #number of event per signal mass point (4)
        self.transform = transform
        self.m_list = m_list

        #get dimensions of image
        one_image = self.hdf5_file[background_dataset][0,:,:,:]
        self.dim = one_image[:,0,0]
        self.depth = one_image[0,0,:]


    def __len__(self):
        return 2*self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_batch = len(idx)
        events = np.zeros(n_batch,self.dim,self.dim,self.depth)
        masses = np.zeros(n_batch)
        labels = np.zeros(n_batch)

        j = 0

        for i in idx:

            if i<self.n:
                #signal event
                group = self.signal_groups[i // self.n_per_m]
                image_index = i % self.n_per_m
                labels[j] = 1
                masses[j] = self.m_list[i // self.n_per_m]

            if i>=self.n:
                group = self.background_group
                image_index = i-self.n
                labels[j] = 0

                m_randint = np.random.randint(self.n_m)
                masses[j] = self.m_list[m_randint] #random mass for background 

            dataset = self.hdf5_file[group] #pointer to dataset in disk
            events[j,:,:,:] = dataset[-image_index,:,:,:] #locally save desired image (from end of dataset to avoid overlapping with training events)

            j+=1

        #convert to tensors
        labels = torch.from_numpy(labels)
        masses = torch.from_numpy(masses).double()

        if self.transform:

            events = self.transform(events) #minimallt convert to tensors

        return (events,labels,masses)



class ConvNet(torch.torch.nn.Module):
    """
    CNN with two conv. layers and 2 fully connected layers to train for single mass point
    Architecture not optimized!
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(5 * 5 * 64, 500)
        self.fc2 = torch.nn.Linear(500, 2)


    def forward(self, x):
    	out = self.layer1(x)
    	out = self.layer2(out)
    	out = out.reshape(out.size(0), -1)
    	out = self.drop_out(out)
    	out = F.relu(self.fc1(out))
    	out = self.fc2(out)
    	return out

class ModelExtraInput(torch.torch.nn.Module):
    """
    Same simple architecture as ConvNet, but take extra input as an argument (A' mass!), 
    adds it the the 2nd FCC layer and has two extra FCC layers to compute an output as 
    a function of mass
    """
    def __init__(self):
        super(ModelExtraInput, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(5 * 5 * 64, 20)
        
        self.fc2 = torch.nn.Linear(20 + 1, 60)
        self.fc3 = torch.nn.Linear(60, 2)
        
    def forward(self, x, m):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        x1 = out.float()
        x2 = torch.empty([len(x),1])
        x2[:,0] = m.float()
        
        out = torch.cat((x1, x2.float()), dim=1)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        


def train_test_cnn_single_m(model,train_loader,test_loader,n_epochs,learn_rate,batch_size,nam,dim,depth=3):
    """
    takes in CNN model who accepts a single mass, the prepared iterators for training and test events,
    the number or epochs for training, the learning rate (constant for now, can be changed), the batch size for
    training, a string to title plots (nam), and the dimensions of the images used for training

    Trains the model on the training data, evaluates the accuracy on the test data at every epoch and prints results

    plots a histogram of the output of the network on the test dataset for signal and for background

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images, in the order in which they were fed to the network
    """
    # Loss and optimizer
    criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(n_epochs):
        for i, (images,labels,masses) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images) #predictions
            labels = labels.long()

            loss = criterion(outputs, labels) #compute loss between output vs actual label
            loss_list.append(loss.item()) #add loss for this epoch to loss list

            # Backprop and perform Adam optimisation
            optimizer.zero_grad() #all gradients equal to 0
            loss.backward() #backpropagation
            optimizer.step() 

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

        model.eval()

        n_test = len(test_loader)*batch_size
     
        label_tab= np.zeros(n_test) #arrays to contain output information
        mod_tab = np.zeros((n_test,2))
        im_tab = np.zeros((n_test,dim,dim,depth))


        with torch.no_grad(): #without modifying any parameters
            correct = 0
            total = 0
            for i,(images, labels,masses) in enumerate(test_loader):

                outputs = model(images) #evaluate output on test data
                mod_outputs = F.softmax(outputs) #convert to probability
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()


                #insert outputs in arrays to plot result of training
                label_tab[i*batch_size:(i+1)*batch_size] = labels.data.numpy()
                mod_tab[i*batch_size:(i+1)*batch_size,:] = mod_outputs.data.numpy()
                im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)

                i+=1

            print('Test Accuracy of the model on the test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = mod_tab[:,1] #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    path_plot = '/plots/'
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    plt.text(0.001,200,'Efficiency = {:.2f}\nm_A = {} MeV\n Batch size = {}\n# Epochs = {}\nLearning rate = {}'.format((correct / total) * 100,100,batch_size,n_epochs,learn_rate))
    plt.legend(loc='upper right')
    plt.title(nam)
    plt.savefig(path_plot+'output_dist.png')

    return mod_tab,label_tab,im_tab


def train_test_cnn_multiple_m(model,train_loader,test_loader,n_epochs,learn_rate,batch_size,nam,dim,depth=3):
    """
    takes in CNN model who accepts mass as a extra input, the prepared iterators for training and test events,
    the number or epochs for training, the learning rate (constant for now, can be changed), the batch size for
    training, a string to title plots (nam), and the dimensions of the images used for training

    Trains the model on the training data, evaluates the accuracy on the test data at every epoch and print results

    plots a histogram of the output of the network on the test dataset for signal and for background

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images, in the order in which they were fed to the network
    """
    # Loss and optimizer

    n_test = len(test_loader)*batch_size
    n_train = len(train_loader)*batch_size
     
    label_tab= np.zeros(n_test+n_train)
    out_tab = np.zeros(n_test+n_train)
    id_tab = np.zeros((n_test+n_train,2))
    m_tab = np.zeros((n_test+n_train))

    criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(n_epochs):
        for g in optimizer.param_groups:
            g['lr'] = g['lr']/2 # reduce learning rate by two every epoch
            print(g['lr'])

        for i, (images,labels,masses) in enumerate(train_loader):
            # Run the forward pass

            outputs = model(images,masses/1000) #predictions
            labels = labels.long()

            loss = criterion(outputs, labels) #compute loss between output vs actual label
            loss_list.append(loss.item()) #add loss for this epoch to loss list

            # Backprop and perform Adam optimisation
            optimizer.zero_grad() #all gradients equal to 0
            loss.backward() #backpropagation
            optimizer.step() 

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


    with torch.no_grad(): #without modifying any parameter
        for i,(images, labels,masses) in enumerate(test_loader):

            outputs = model(images,masses/1000)
            mod_outputs = F.softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            #insert outputs in arrays to plot result of training
             
            label_tab[n_train+i*batch_size:n_train+(i+1)*batch_size] = labels.data.numpy()
            out_tab[n_train+i*batch_size:n_train+(i+1)*batch_size] = mod_outputs.data.numpy()[:,1]
            im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)
            m_tab[n_train+i*batch_size:n_train+(i+1)*batch_size] = masses.data.numpy()
            


        print('Test Accuracy of the model on test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = out_tab #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    path_plot = '/plots/'
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    plt.text(0.001,200,'Efficiency = {:.2f}\nm_A = {} MeV\n Batch size = {}\n# Epochs = {}\nLearning rate = {}'.format((correct / total) * 100,100,batch_size,n_epochs,learn_rate))
    plt.legend(loc='upper right')
    plt.title(nam)
    #plt.savefig(path_plot+'output_dist.png')
    plt.show()

    return out_tab,label_tab,im_tab,m_tab

def test_any_model(model,test_loader,nam,batch_size,dim,depth=3):
    """
    takes in pre-trained CNN model who accepts mass as a extra input, the prepared iterator for test events,
    the batch size, a string to title plots (nam), and the dimensions of the images used for training

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images and array with tags to find original events, in the order in which they were fed to the network
    """
    # Loss and optimizer
  
    model.eval()

    n_test = len(test_loader)*batch_size
     
    label_tab= np.zeros(n_test)
    out_tab = np.zeros(n_test)
    im_tab = np.zeros((n_test,dim,dim,depth))
    id_tab = np.zeros((n_test,2))
    m_tab = np.zeros((n_test))


    with torch.no_grad(): #without modifying any parameter
        correct = 0
        total = 0
        for i,(images, labels,masses) in enumerate(test_loader):

            outputs = model(images,masses/1000)
            mod_outputs = F.softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            #insert outputs in arrays to plot result of training
             
            label_tab[i*batch_size:(i+1)*batch_size] = labels.data.numpy()
            out_tab[i*batch_size:(i+1)*batch_size] = mod_outputs.data.numpy()[:,1]
            im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)
            m_tab[i*batch_size:(i+1)*batch_size] = masses.data.numpy()
            i+=1


        print('Test Accuracy of the model on the test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = out_tab #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    #bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    """
    path_plot = '/plots/'
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    #plt.text(0.0001,1000,'Efficiency = {:.2f}\nm_A = {} MeV'.format((correct / total) * 100,m_tab[0]))
    plt.legend(loc='upper left')
    plt.ylim([0.1,1e5])
    plt.title(nam)
    #plt.savefig(path_plot+'output_dist.png')
    plt.show()
    """

    return out_tab,label_tab,im_tab,m_tab


def roc_curve(labels,output,batch_size,n_epochs,learning_rate,nam):
    """
    takes in array of labels (targets) and corresponding array of network output
    (probability of an event being signal) for test events.
    Produces ROC curve and computes area under
    Returns area under ROC and threshold for signal minimizing the number of misclassified events
    """
    n_p = 500 #number of points to plot ROC curve

    thresh_tab = np.flip(np.linspace(0,1,n_p))

    tvp = np.zeros(n_p) #to contains true positive rate
    tfp = np.zeros(n_p) #to contain false positive rate
    misc = np.zeros(n_p) #to contain number of misclassified events
    t_p_frac = np.zeros(n_p) #to contain number of misclassified events passing the cut


    for i,t in enumerate(thresh_tab):
        positives = (output>t)

        vp_ind = np.where(positives*labels>0)[0] #events where true positive
        n_vp = len(vp_ind) #number of true positives

        fp_ind = np.where(positives*(labels+1)==1)[0] #where false positives
        n_fp = len(fp_ind)

        vn_ind = np.where((positives+labels)==0)[0] #where true negatives
        n_vn = len(vn_ind)

        fn_ind = np.where((positives+1)*labels==1)[0] #where false negatives
        n_fn = len(fn_ind)

        tvp[i] = n_vp/(n_vp+n_fn)

        tfp[i] = n_fp/(n_fp+n_vn)

        misc[i] = n_fp + n_fn
        if (n_fp + n_vp)!=0:
        	t_p_frac[i] = n_vp/(n_fp + n_vp)
        else:
        	t_p_frac[i] = 0

    area_under = np.trapz(tvp,tfp) #computes area under curve using trap. rule

    #define suggested threshold where it minimises misclassified events (change?)
    #threshold = thresh_tab[np.argmin(misc)]

    #define threshold for signal classification where it allows rejection of 99.9% of bkgd to compare to BDT
    ind_back = np.where(labels == 0)[0]
    out_back = np.sort(output[ind_back]) #outputs of background
    frac = int(len(out_back)/1000)
    threshold = out_back[-frac]

    print('optimal threshold = {:.3f}'.format(threshold))
    print('AU ROC = {:.4f}'.format(area_under))

    #plot ROC curve
    path_plot = '/plots/'
    plt.plot(tfp,tvp,'k-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-2,1.1])
    plt.title(nam)
    plt.text(0.2,0.8,'AU ROC = {:.4f}\nOpt. thresh. = {:.2f}\nBatch size = {}\n# Epochs = {}\nLearning rate = {}'.format(area_under,threshold,batch_size,n_epochs,learning_rate))
    #plt.savefig(path_plot+'roc_curve.png')
    plt.show()

    return area_under, threshold



def accuracy(labels, outs, thresh):
    """
    Function returning accuracy over all events, acceptance of signal, portion of events passing cut being signal
    and background suppression fraction
    """
    classification = (outs>thresh) #final classification from this network, 0 if background, 1 if signal
    n_tot = len(classification)

    n_false = len(np.where((classification+labels)==1)[0]) #number of misc. events.

    n_sig = np.sum(labels) #actual number of signal events
    n_back = n_tot - n_sig #background events

    sig_false = len(np.where((classification+1)*labels==1)[0]) #number of misc. sig events
    sig_true = n_sig - sig_false #number of properly classied signal events
    back_false = n_false - sig_false # number of background events passing cuts
    back_true = (n_back - back_false)/n_back #properly classified background

    accur = np.sum(classification==labels)/n_tot #fraction of events classified correctly
    accept = sig_true/n_sig
    sign_frac = sig_true/(back_false+sig_true)
    back_sup = back_true

    return accur,accept,sign_frac,back_sup



def mean_per_channel(image_array):
    
    #Mean and std for each channel of pictures for dataset normalisation (bot used anymore, makes
    #performance worse) to feed in as argument to transforms.Normalize
    

    num_channel = len(image_array[0,0,0,:])
    mean_tab = []
    std_tab= []
    for c in range(num_channel):
        mean_tab.append(np.mean(image_array[:,:,:,c]))
        std_tab.append(np.std(image_array[:,:,:,c]))

    max_v = np.amax(image_array)
    return max_v,tuple(mean_tab),tuple(std_tab)

