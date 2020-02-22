#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
format data before training
Andrea Gaspert 02/2020 agaspert@stanford.edu
"""
import matplotlib
#matplotlib.use('Agg') #if running on cluster and cannot have gui to produce figures

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import time

#maxima of x,y and z positions in Ecal as defined in outputs of sim., to define lattice params
x_min, x_max = -200,200
y_min, y_max = -200,200
z_min, z_max = -200,200


def total_hits(npy_array):
	"""
	Counts total number of event and Ecal hits in npy array converted from ROOT
	"""
	n_event = np.size(npy_array)

	n_tot = 0

	for j in range(n_event): 
		n_tot+=npy_array[j][1]

	return n_event, n_tot



def plot_scatter(arr2,depth, out=0,title=' ',original=False,arr1=[]):
	#function to plot original hits in detector vs what is used for training

	if original:

		indexes = np.argsort(arr1['e_tab'])

		n_hits = len(arr1['e_tab'])
		arr1 = arr1[indexes]

		fig = plt.figure()
		ax = fig.add_subplot(111,projection = '3d')

		e_max = np.amax(arr1['e_tab'])
		m_size = arr1['e_tab']/e_max*20
		m_size = m_size.astype(int)


		ax.scatter(arr1['z_tab'], arr1['x_tab'], arr1['y_tab'], s=m_size, c=arr1['e_tab'],vmin=0,vmax=e_max,cmap='jet')

		ax.set_xlabel('Z')
		ax.set_ylabel('X')
		ax.set_zlabel('Y')
		plt.title(title)
		plt.show()

	rows = int(depth/2) +1
	fig, axs = plt.subplots(rows, 2)

	k=0

	for i in range(rows):
		for j in range(2):
			axs[i, j].matshow(arr2[:,:,k])
			axs[i, j].set_title('Channel {}'.format(k))
			k+=1
			if k>=depth:
				break
	for ax in axs.flat:
		ax.label_outer()
	if out>0:
		plt.title('output={}'.format(out))
	plt.show()



def im_split_z(data,dim,depth):
	"""
	function accepts lists of arrays with data ([x,y,z,e]) and returns images for training for every EVENT (not hit)
	dim = dimension of image wanted for training (i.e. will produce a dim x dim output array)
	depth = number of channel for image. Z axis will be split into depth parts and each part will be 
	summed and added to a channel
	"""

	n_tot = len(data[0][:,0]) #number of events

	#define lattice to insert energies into to creta images
	x_ref = np.linspace(x_min,x_max,dim)
	y_ref = np.linspace(y_min,y_max,dim)
	z_ref = np.linspace(z_min,z_max,depth)

	h=0 #count events
	im_array = np.zeros(n_tot,dim,dim,depth)

	while h<n_tot: #loop over events

		im_tab = np.zeros((dim,dim,depth)) #array to contain constructed image

		#remove empty hits
		ind_nul = np.where(data[3][h,:]==0)[0]
		x_tab = data[0][h,:ind_nul]
		y_tab = data[1][h,:ind_nul]
		z_tab = data[2][h,:ind_nul]
		e_tab = data[3][h,:ind_nul]

		for i in range(len(e_tab)): #loop over hits within event

			x_coord = np.argmin(np.abs(x_ref-x_tab[i]))
			y_coord = np.argmin(np.abs(y_ref-y_tab[i]))
			z_coord = np.argmin(np.abs(z_ref-z_tab[i]))

			im_array[h,x_coord,y_coord,z_coord] += e_tab[i]


		#show a few plots of actual event vs events converted to picture
		"""
		if h<5:
			data_actual = [x_tab,y_tab,z_tab,e_tab]
			data_comp = im_array[h,:,:,:]
			plot_scatter(data_actual,data_comp,depth)
		"""
		h+=1 #next event

		if h % 10000 == 0:
			print('{}% completed'.format(h/n_tot*100))

	return im_array


def im_xyz(data,dim,depth=3):
	#function accepting list of hits and returning images for training for every EVENT (not hit)

	n_tot = len(data[0][:,0]) #number of events

	x_ref = np.linspace(x_min,x_max,dim) #definition of lattice points for creation of images
	y_ref = np.linspace(y_min,y_max,dim)
	z_ref = np.linspace(z_min,z_max,dim)


	h=0 #count events
	im_array = np.zeros(n_tot,dim,dim,3)


	while h<n_tot: #loop over events

		ind_nul = np.where(data[3][h,:]==0)[0][0]
		x_tab = data[0][h,:ind_nul]
		y_tab = data[1][h,:ind_nul]
		z_tab = data[2][h,:ind_nul]
		e_tab = data[3][h,:ind_nul]

		for i in range(len(e_tab)):
			x_coord = np.argmin(np.abs(x_tab[i]-x_ref)) #coordinates of pixel where energy for this hit will be added
			y_coord = np.argmin(np.abs(y_tab[i]-y_ref))
			z_coord = np.argmin(np.abs(z_tab[i]-z_ref))

			im_array[h,x_coord,y_coord,0] += e_tab[i] #add energy to proper pixels
			im_array[h,y_coord,z_coord,1] += e_tab[i]
			im_array[h,z_coord,x_coord,2] += e_tab[i]

		#show a few plots of actual event vs events converted to picture
		"""
		if h<5:
			data_actual = [x_tab,y_tab,z_tab,e_tab]
			data_comp = im_array[h,:,:,:]
			plot_scatter(data_actual,data_comp,depth)
		"""

		h+=1 #next event

		if h % 10000 == 0:
			print('{}% completed'.format(h/n_tot*100))

	return im_array

def prep_data(data_group,im_dataset_name,format_type,dim,depth=3):

	#intakes name of data group in hdf5 file and creates new dataset in group containing images for training, 
	#given desired parameters

	if format_type == 'z_split':
		fun = im_split_z
	if format_type == 'xyz':
		fun = im_xyz

	with h5py.File('dataset.hdf5', 'r') as f:
		#read list of hits
		data_x = f.get(data_goup+'/hits_x') 
		data_y = f.get(data_goup+'/hits_y')
		data_z = f.get(data_goup+'/hits_z')
		data_e = f.get(data_goup+'/hits_e')

		im_array =  fun([data_x,data_y,data_z,data_e],dim,depth)


	with h5py.File('dataset.hdf5', 'a') as f:
		#append file with array of images
		new_data = f.create_dataset(data_goup+'/'+im_dataset_name,data=im_array)
		new_data.attrs['date'] = time.time()
