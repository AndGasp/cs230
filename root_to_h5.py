#!/usr/bin/env python                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                  

import ROOT
from sys import exit
import os
import datetime

try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()

try:
    import h5py
except:
    print("Failed to import h5py.")
    exit()

ecal_attributes = ['eventNumber', 'nHits', 'nNonPositiveHits',
'BDTval', 'nReadoutHits', 'totEdepECal', 'totEdepECalTightIso', 'maxCellEdepEcal', 
'deepestHitLayer', 'averageHitLayer', 'stdHitLayer', 'showerRMS', 'stdX', 'stdY', 
'ecalBackEnergy', 'electronContainmentEnergy', 'photonContainmentEnergy', 
'outsideContainmentEnergy', 'outsideContainmentNHits', 'outsideContainmentXStd', 
'outsideContainmentYStd', 'passEcal', 'passHcal', 'electron_energy', 'electron_pT', 
'electron_phi', 'electron_theta'] #list of attributes one can plot

def root_to_h5(root_file,dataset_name,extra=True):
        """                                                                                                                                              
        Read root file and converts tree to structured array                                                                                             
        one entry for each event, fields: 'x_tab', 'y_tab', 'z_tab' and 'e_tab'                                                                          
        are arrays of the length of th number of hits per events and 'id' is the number
        of the event in the original root file    
        If extra = true, also creates and saves other structured array with other info about events                                                                                    
        """
        if 'ecal' in root_file:

            if 'EOT' in root_file:
                path_in = '/nfs/slac/g/ldmx/users/lene/hits/4gev_1e_ecal_pn_00_3.21e13/'
                path_out = '/nfs/slac/g/ldmx/users/lene/hits/Formatted/root_2_array/ecal/'
                group_name = 'background/ecal'

            if 'mA' in root_file:
                path_in = '/nfs/slac/g/ldmx/users/lene/hits/'                                                             
                path_out = '/nfs/slac/g/ldmx/users/lene/hits/Formatted/root_2_array/ecal/'
                group_name = 'signal/ecal'

            f = ROOT.TFile.Open(path_in+root_file)
            tree = f.Get('ecalHits')

            file_un = ROOT.TFile(path_in+root_file)
            reader = ROOT.TTreeReader('ecalHits',file_un) #read events

        if 'hcal' in root_file:
            if 'EOT' in root_file:
                path_in = '/nfs/slac/g/ldmx/users/lene/hits/4gev_1e_ecal_pn_00_3.21e13/'
                path_out = '/nfs/slac/g/ldmx/users/lene/hits/Formatted/root_2_array/hcal/'
                group_name = 'background/hcal'

            if 'mA' in root_file:
                path_in = '/nfs/slac/g/ldmx/users/lene/hits/'#WATCHOUT!!!                                                             
                path_out = '/nfs/slac/g/ldmx/users/lene/hits/Formatted/root_2_array/hcal'
                group_name = 'signal/hcal'

            f = ROOT.TFile.Open(path_in+root_file)
            tree = f.Get('hcalHits')

            file_un = ROOT.TFile(path_in+root_file)
            reader = ROOT.TTreeReader('hcalHits',file_un) #read events

        n_tot = int(tree.GetEntries()) #number of events 

        #print branch names
        branch_list = tree.GetListOfBranches()
        all_branch_names = [branch_list.At(i).GetName() for i in range(branch_list.GetEntries())]
        #print(all_branch_names)


        #Create numpy sturctured array with extra info about events=======================================
        extra_labels = ['eventNumber', 'nHits', 'nNonPositiveHits'].append(all_branch_names[7:])

        data_extra = np.zeros((n_tot,len(ecal_attributes)))

        #create list of readers for extra variables:
        readers_extra = [ROOT.TTreeReaderValue(float)(reader, extra_labels[j]) for j in range(len(extra_labels))]

        max_hits = 0 #maximal number of hits in a single event in this tree                                                                         
        event_ids = [] #event number within file to keep track of events 
        n_hit = [] #number of hits for every event

        for i,event in enumerate(tree):
            event_ids.append(event.eventNumber) #ids of events in the file to keep track
            event_hits  = event.nHits #number of hits in this event
            n_hit.append(event_hits) #list to contain all number of hits for every event

            if event_hits > max_hits:
                tot_event = event_hits #maximal number of hits of any event in this file
       
        print(n_tot,tot_event)

        #create arrays to contain info for training
        data_x = np.zeros((n_tot,tot_event))
        data_y = np.zeros((n_tot,tot_event))
        data_z = np.zeros((n_tot,tot_event))
        data_E = np.zeros((n_tot,tot_event))


		#readers to loop over arrays of x,y,z and E                                                                                                     
        x_r = ROOT.TTreeReaderArray(float)(reader, 'x')
        y_r = ROOT.TTreeReaderArray(float)(reader, 'y')
        z_r = ROOT.TTreeReaderArray(float)(reader, 'z')
        E_r = ROOT.TTreeReaderArray(float)(reader, 'E')

        #create list of readers for extra variables:
        if extra:
            #list of readers for extra values, assumes all extra branches contain values (not Arrays!!!)
            #does not work because not all are floats! Figure out way to adapt variable type
            readers_extra = [ROOT.TTreeReaderValue(float)(reader, extra_labels[j]) for j in range(len(extra_labels))]

        j = 0 #loop over events                                                                                                                          
        while reader.Next():
            h = n_hit[j] #number of hits for this event

            for k in range(len(extra_labels)):
                #fill extra data array with extra data
                data_extra[j,k] = readers_extra[k]

            x = [] #to contain list of info for different hits within one event
            y = []
            z = []
            E = []
            for i in range(len(x_r)):  
            	#loop over hits within events and store values in arrays                                                                                    
                x.append(float(x_r[i]))
                y.append(float(y_r[i]))
                z.append(float(z_r[i]))
                E.append(float(E_r[i]))

            #print(x)
            #print(e)

            if extra:
            	data_extra[j]['file_id'] = id_name
            	data_extra[j]['event_id'] = event_ids[j]
            	for k in range(len(extra_labels)):
            		#fill extra data array with extra data
            		data_extra[j][extra_labels[k]] = readers_extra[k]


            data_x[j,:h] = np.array(x)
            data_y[j,:h] = np.array(y)
            data_z[j,:h] = np.array(z)
            data_E[j,:h] = np.array(E)
            j += 1

            if j%10000 == 0:
                print(j*100/n_tot)

                                                                                                                      
        #check if hdf5 file exists, create if not                                                                                                                       
        if not os.path.exists('dataset.hdf5'):                                                                                                                                            
            print('Creating hdf5 file to contain data')
            f = h5py.File('dataset.hdf5', 'w')
            g1 = f.create_group('signal')
            gg11 = g1.create_group('ecal')
            gg12 = g1.create_group('hcal')
            g2 = f.create_group('background') 
            gg21 = g2.create_group('ecal')
            gg22 = g2.create_group('hcal')
        #open for modification if it exists
        else: 
            f = h5py.File('dataset.hdf5','a')

        #create new sub-group for new data in file
        new_group = group_name+'/'+dataset_name
        newg = hf.create_group(new_group)

        #insert datasets in group
        newg.create_dataset('hits_E',data=data_E)
        newg.create_dataset('hits_x',data=data_x)
        newg.create_dataset('hits_y',data=data_y)
        newg.create_dataset('hits_z',data=data_z)
        newg.create_dataset('attr',data=data_extra)

        #add attributes with dataset info
        newg.attrs['date'] = time.time()
        newg.attrs['source'] = dataset_name

        f.close()





#add function to overwrite root files with versions with output results

#add function to create histograms of all events (bdt vs network, one variable histogram, correlation between two variables?)
#run over all necessary files

f_names = ['ntuple_ecal_hits_1.8e8EOT_{}.root'.format(i) for i in range(50)] #run over ecal background events
dataset_names = ['ecal_back_{}'.format(i) for i in range(50)]

for i,f in enumerate(f_names):
    root_to_h5(f,dataset_names)

f_names = ['ecalHits_signal_my-mA1MeV.root','ecalHits_signal_my-mA5MeV.root','ecalHits_signal_my-mA10MeV.root','ecalHits_signal_my-mA50MeV.root',
'ecalHits_signal_my-mA100MeV.root','ecalHits_signal_my-mA500MeV.root','ecalHits_signal_my-mA1000MeV.root'] #run over ecal signal events
dataset_names = ['ecal_signal_1','ecal_signal_5','ecal_signal_10','ecal_signal_50','ecal_signal_100','ecal_signal_500','ecal_signal_1000']

for i,f in enumerate(f_names):
    root_to_h5(f,dataset_names)