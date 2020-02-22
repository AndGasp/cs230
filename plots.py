import numpy as np
import matplotlib.pyplot as plt



def plot_hist_2d(list1,list2,var1,var2,limy=[0,1],binx=16,biny=16):


    #gridspec.GridSpec(2,2)
    #plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    plt.clf()
    plt.ylabel(var1, fontsize=18) #nbins=
    plt.xlabel(var2, fontsize=18)
    #plt.title(name, fontsize=20)
    H, yedges, xedges = np.histogram2d(list1, list2, bins=[binx,biny], range=[[0,1],limy])
    Hmasked = np.ma.masked_where(H==0,H) #cases vides blanches
    #print(type(H),np.shape(H))
    for i in range(np.size(Hmasked[0,:])):
        if np.max(Hmasked[i,:]) != 0:
            Hmasked[i,:]=Hmasked[i,:]/np.max(Hmasked[i,:]) #normaliser chaque rangée de l'histogramme
        for j in range(np.size(Hmasked[:,0])):
            if Hmasked[i,j]<0:
                #print('valeur negative! ({}): ({},{})'.format(Hmasked[i,j],i,j))
                Hmasked[i,j]=0 #mettre valeurs négatives à 0 (artéfact calcul de section efficace)

    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.tick_params(labelsize=16)

    #cbar = plt.colorbar()
    #cbar.ax.set_ylabel('Pr({}|BDT Output)'.format(var2), fontsize=20)  

    plt.show()

def plot_hist(listb,lists1,lists2,var1,w1,limi=[0,1],bins=40):
        bins = np.linspace(limi[0],limi[1], bins)

        plt.ylabel('Counts', fontsize=18)
        #plt.title('Distribution weighed by CNN output', fontsize=18)
        plt.hist(listb, bins, weights=w1, normed=1, alpha=0.5, label='Background')
        plt.hist(lists1, bins, weights=w1, normed=1, alpha=0.5, label='Signal-1 MeV')
        plt.hist(lists2, bins, weights=w1, normed=1, alpha=0.5, label='Signal-1000 MeV')
        plt.tick_params(labelsize=16)


        plt.legend(loc='upper right',fontsize=12)
        plt.xlabel(var1,fontsize=12)
        plt.ylabel('Events',fontsize=12)

        plt.yscale('log')

        plt.show()

def plot_sb(list1,var1,w1,limi=[0,1],bins=40,title=''):
        bins = np.linspace(limi[0],limi[1], bins)

        plt.ylabel('Counts', fontsize=18)
        #plt.title('Distribution weighed by CNN output', fontsize=18)
        plt.hist(list1, bins, weights=w1, normed=1, alpha=0.5, label='Signal-like')
        plt.hist(list1, bins, weights=1-w1, normed=1, alpha=0.5, label='Background-like')
        plt.tick_params(labelsize=16)


        plt.legend(loc='upper right',fontsize=12)
        plt.xlabel(var1,fontsize=12)
        plt.ylabel('Events',fontsize=12)
        plt.title(title)

        plt.yscale('log')

        plt.show()


extra_b = np.load('extra_data/extra_b_26.npy')[-100000:]
extra_s1 = np.load('extra_data/extra_s_1.npy')[-100000:]
extra_s5 = np.load('extra_data/extra_s_5.npy')[-100000:]
extra_s10 = np.load('extra_data/extra_s_10.npy')[-100000:]
extra_s50 = np.load('extra_data/extra_s_50.npy')[-100000:]
extra_s100 = np.load('extra_data/extra_s_100.npy')[-100000:]
extra_s500 = np.load('extra_data/extra_s_500.npy')[-100000:]
extra_s1000 = np.load('extra_data/extra_s_1000.npy')[-100000:]
"""
p_t = np.concatenate((extra_b[:,16],extra_s1[:,24],extra_s10[:,24],extra_s100[:,24],extra_s1000[:,24]))
bdt = np.concatenate((extra_b[:,2],extra_s1[:,3],extra_s10[:,3],extra_s100[:,3],extra_s1000[:,3]))
ave_hit_layer = np.concatenate((extra_b[:,8],extra_s1[:,9],extra_s10[:,9],extra_s100[:,9],extra_s1000[:,9]))
tot_e_dep = np.concatenate((extra_b[:,4],extra_s1[:,5],extra_s10[:,5],extra_s100[:,5],extra_s1000[:,5]))
n_hit = np.concatenate((extra_b[:,1],extra_s1[:,1],extra_s10[:,1],extra_s100[:,1],extra_s1000[:,1]))
output_cnn = np.concatenate((results[:,0],results[:,1],results[:,3],results[:,5],results[:,7]))
"""
w1 = np.ones(100000)
"""
#actual distributions
#nHits
plot_hist(extra_b[:,1],extra_s1[:,1],extra_s1000[:,1],'nHits',w1,limi=[0,200],bins=100)
#plot_hist(extra_b[:,1],extra_s50[:,1],'nHits',w1,limi=[0,500],bins=100)


#totEdepEcal
plot_hist(extra_b[:,4],extra_s1[:,5],extra_s1000[:,5],'totEdepEcal',w1,limi=[0,4000],bins=100)
#plot_hist(extra_b[:,4],extra_s50[:,5],'totEdepEcal',w1,limi=[0,500],bins=100)
"""
#totEdepEcal
plot_hist(extra_b[:,5],extra_s1[:,6],extra_s1000[:,6],'totEdepEcal tight',w1,limi=[0,20000],bins=100)
#plot_hist(extra_b[:,5],extra_s50[:,6],'totEdepEcal tight',w1,limi=[0,500],bins=100)

#max cell Edep Ecal
plot_hist(extra_b[:,6],extra_s1[:,7],extra_s1000[:,7],'maxCell EdepEcal',w1,limi=[0,10000],bins=60)
#plot_hist(extra_b[:,6],extra_s50[:,7],'maxCell EdepEcal',w1,limi=[0,100],bins=40)

"""
#deepest hit layer
plot_hist(extra_b[:,7],extra_s1[:,8],extra_s1000[:,8],'Deepest hit layer',w1,limi=[0,50],bins=50)
#plot_hist(extra_b[:,7],extra_s50[:,8],'Deepest hit layer',w1,w2,limi=[0,30],bins=30)


#average hit layer
plot_hist(extra_b[:,8],extra_s1[:,9],extra_s1000[:,9],'ave. hit layer',w1,limi=[0,30],bins=30)
#plot_hist(extra_b[:,8],extra_s50[:,9],'ave. hit layer',w1,w2,limi=[0,30],bins=30)


#std hit layer
plot_hist(extra_b[:,9],extra_s1[:,10],extra_s1000[:,10],'std hit layer',w1,limi=[0,15],bins=15)
#plot_hist(extra_b[:,9],extra_s50[:,10],'std hit layer',w1,w2,limi=[0,30],bins=30)

#shower rms
plot_hist(extra_b[:,10],extra_s1[:,11],extra_s1000[:,11],'shower rms',w1,limi=[0,50],bins=50)
#plot_hist(extra_b[:,10],extra_s50[:,11],'shower rms',w1,w2,limi=[0,30],bins=30)


#stdX
plot_hist(extra_b[:,11],extra_s1[:,12],extra_s1000[:,12],'stdX',w1,limi=[0,50],bins=50)
#plot_hist(extra_b[:,11],extra_s50[:,12],'stdX',w1,w2,limi=[0,30],bins=30)


#stY
plot_hist(extra_b[:,12],extra_s1[:,13],extra_s1000[:,13],'stdX',w1,limi=[0,50],bins=50)
#plot_hist(extra_b[:,12],extra_s50[:,13],'stdX',w1,w2,limi=[0,30],bins=30)


#pass hcal veto
plot_hist(extra_b[:,13],extra_s1[:,21],extra_s1000[:,21],'pass ecal veto',w1,limi=[-0.5,1.5],bins=2)
#plot_hist(extra_b[:,13],extra_s50[:,21],'pass ecal veto',w1,w2,limi=[0,1],bins=2)


#pass ecal veto
plot_hist(extra_b[:,14],extra_s1[:,22],extra_s1000[:,22],'pass Hcal veto',w1,limi=[-0.5,1.5],bins=2)
#plot_hist(extra_b[:,14],extra_s50[:,22],'pass Hcal veto',w1,w2,limi=[0,1],bins=2)


#e energy
plot_hist(extra_b[:,15],extra_s1[:,23],extra_s1000[:,23],'e energy',w1,limi=[0,5000],bins=100)
#plot_hist(extra_b[:,15],extra_s50[:,23],'e energy',w1,w2,limi=[0,100],bins=40)


#e pt
plot_hist(extra_b[:,16],extra_s1[:,24],extra_s1000[:,24],'e Pt',w1,limi=[0,1000],bins=100)
#plot_hist(extra_b[:,16],extra_s50[:,24],'e Pt',w1,w2,limi=[0,100],bins=40)


#e phi
plot_hist(extra_b[:,17],extra_s1[:,25],extra_s1000[:,25],'e phi',w1,limi=[0,3.2],bins=20)
#plot_hist(extra_b[:,17],extra_s50[:,25],'e phi',w1,w2,limi=[0,5],bins=20)


#e theta
plot_hist(extra_b[:,18],extra_s1[:,26],extra_s1000[:,26],'e theta',w1,limi=[0,1.6],bins=20)
#plot_hist(extra_b[:,18],extra_s50[:,26],'e theta',w1,w2,limi=[0,5],bins=20)
"""

#how BDT performs for all variables for 1, 50 and 1000 MeV signal
"""
#nHits
plot_sb(extra_s1[:,1],'nHits',extra_s1[:,3],limi=[0,200],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,1],'nHits',extra_s1000[:,3],limi=[0,200],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,1],'nHits',extra_s50[:,3],limi=[0,200],bins=100,title='BDT, m=50 MeV')

#totEdepEcal
plot_sb(extra_s1[:,5],'totEdep',extra_s1[:,3],limi=[0,5000],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,5],'totEdep',extra_s1000[:,3],limi=[0,5000],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,5],'totEdep',extra_s50[:,3],limi=[0,5000],bins=100,title='BDT, m=50 MeV')

#totEdepEcal
plot_sb(extra_s1[:,6],'totEdep tight',extra_s1[:,3],limi=[0,5000],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,6],'totEdep tight',extra_s1000[:,3],limi=[0,5000],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,6],'totEdep tight',extra_s50[:,3],limi=[0,5000],bins=100,title='BDT, m=50 MeV')

#max cell Edep Ecal
plot_sb(extra_s1[:,7],'max cell Edep',extra_s1[:,3],limi=[0,1000],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,7],'max cell Edep',extra_s1000[:,3],limi=[0,1000],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,7],'max cell Edep',extra_s50[:,3],limi=[0,1000],bins=100,title='BDT, m=50 MeV')


#deepest hit layer
plot_sb(extra_s1[:,8],'Deepest hit',extra_s1[:,3],limi=[0,50],bins=50,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,8],'Deepest hit',extra_s1000[:,3],limi=[0,50],bins=50,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,8],'Deepest hit',extra_s50[:,3],limi=[0,50],bins=50,title='BDT, m=50 MeV')


#average hit layer
plot_sb(extra_s1[:,9],'Ave hit',extra_s1[:,3],limi=[0,30],bins=30,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,9],'Ave hit',extra_s1000[:,3],limi=[0,30],bins=30,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,9],'Ave hit',extra_s50[:,3],limi=[0,30],bins=30,title='BDT, m=50 MeV')


#std hit layer
plot_sb(extra_s1[:,10],'std hit',extra_s1[:,3],limi=[0,16],bins=16,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,10],'std hit',extra_s1000[:,3],limi=[0,16],bins=16,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,10],'std hit',extra_s50[:,3],limi=[0,16],bins=16,title='BDT, m=50 MeV')

#shower rms
plot_sb(extra_s1[:,11],'shower RMS',extra_s1[:,3],limi=[0,50],bins=50,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,11],'shower RMS',extra_s1000[:,3],limi=[0,50],bins=50,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,11],'shower RMS',extra_s50[:,3],limi=[0,50],bins=50,title='BDT, m=50 MeV')


#stdX
plot_sb(extra_s1[:,12],'std x',extra_s1[:,3],limi=[0,50],bins=50,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,12],'std x',extra_s1000[:,3],limi=[0,50],bins=50,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,12],'std x',extra_s50[:,3],limi=[0,50],bins=50,title='BDT, m=50 MeV')


#stdY
plot_sb(extra_s1[:,13],'std y',extra_s1[:,3],limi=[0,50],bins=50,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,13],'std y',extra_s1000[:,3],limi=[0,50],bins=50,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,13],'std y',extra_s50[:,3],limi=[0,50],bins=50,title='BDT, m=50 MeV')


#e energy
plot_sb(extra_s1[:,23],'e energy',extra_s1[:,3],limi=[0,5000],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,23],'e energy',extra_s1000[:,3],limi=[0,5000],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,23],'e energy',extra_s50[:,3],limi=[0,5000],bins=100,title='BDT, m=50 MeV')


#e pt
plot_sb(extra_s1[:,24],'e pt',extra_s1[:,3],limi=[0,1000],bins=100,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,24],'e pt',extra_s1000[:,3],limi=[0,1000],bins=100,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,24],'e pt',extra_s50[:,3],limi=[0,1000],bins=100,title='BDT, m=50 MeV')


#e phi
plot_sb(extra_s1[:,25],'e phi',extra_s1[:,3],limi=[0,3.2],bins=40,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,25],'e phi',extra_s1000[:,3],limi=[0,3.2],bins=40,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,25],'e phi',extra_s50[:,3],limi=[0,3.2],bins=40,title='BDT, m=50 MeV')


#e theta
plot_sb(extra_s1[:,26],'e theta',extra_s1[:,3],limi=[0,1.6],bins=40,title='BDT, m=1 MeV')
plot_sb(extra_s1000[:,26],'e theta',extra_s1000[:,3],limi=[0,1.6],bins=40,title='BDT, m=1000 MeV')
plot_sb(extra_s50[:,26],'e theta',extra_s50[:,3],limi=[0,1.6],bins=40,title='BDT, m=50 MeV')
"""
#how CNN performs for all variables for 1, 50 and 1000 MeV signal

results_s = np.load('results_signal.npy')
w1 = results_s[:,1]
w50 = results_s[:,4]
w1000 = results_s[:,7]

#nHits
#plot_sb(extra_s1[:,1],'nHits',w1,limi=[0,200],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,1],'nHits',w1000,limi=[0,200],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,1],'nHits',w50,limi=[0,200],bins=100,title='CNN, m=50 MeV')

#totEdepEcal
#plot_sb(extra_s1[:,5],'totEdep',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,5],'totEdep',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,5],'totEdep',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')

#totEdepEcal
#plot_sb(extra_s1[:,6],'totEdep tight',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,6],'totEdep tight',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,6],'totEdep tight',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')

#max cell Edep Ecal
#plot_sb(extra_s1[:,7],'max cell Edep',w1,limi=[0,1000],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,7],'max cell Edep',w1000,limi=[0,1000],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,7],'max cell Edep',w50,limi=[0,1000],bins=100,title='CNN, m=50 MeV')


#deepest hit layer
#plot_sb(extra_s1[:,8],'Deepest hit',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,8],'Deepest hit',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,8],'Deepest hit',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#average hit layer
#plot_sb(extra_s1[:,9],'Ave hit',w1,limi=[0,30],bins=30,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,9],'Ave hit',w1000,limi=[0,30],bins=30,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,9],'Ave hit',w50,limi=[0,30],bins=30,title='CNN, m=50 MeV')


#std hit layer
#plot_sb(extra_s1[:,10],'std hit',w1,limi=[0,16],bins=16,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,10],'std hit',w1000,limi=[0,16],bins=16,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,10],'std hit',w50,limi=[0,16],bins=16,title='CNN, m=50 MeV')

#shower rms
#plot_sb(extra_s1[:,11],'shower RMS',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,11],'shower RMS',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,11],'shower RMS',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#stdX
#plot_sb(extra_s1[:,12],'std x',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,12],'std x',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,12],'std x',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#stdY
#plot_sb(extra_s1[:,13],'std y',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,13],'std y',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,13],'std y',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#e energy
#plot_sb(extra_s1[:,23],'e energy',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,23],'e energy',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,23],'e energy',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')


#e pt
#plot_sb(extra_s1[:,24],'e pt',w1,limi=[0,1000],bins=100,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,24],'e pt',w1000,limi=[0,1000],bins=100,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,24],'e pt',w50,limi=[0,1000],bins=100,title='CNN, m=50 MeV')


#e phi
#plot_sb(extra_s1[:,25],'e phi',w1,limi=[0,3.2],bins=40,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,25],'e phi',w1000,limi=[0,3.2],bins=40,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,25],'e phi',w50,limi=[0,3.2],bins=40,title='CNN, m=50 MeV')


#e theta
#plot_sb(extra_s1[:,26],'e theta',w1,limi=[0,1.6],bins=40,title='CNN, m=1 MeV')
#plot_sb(extra_s1000[:,26],'e theta',w1000,limi=[0,1.6],bins=40,title='CNN, m=1000 MeV')
plot_sb(extra_s50[:,26],'e theta',w50,limi=[0,1.6],bins=40,title='CNN, m=50 MeV')

#how CNN performs background (info learned about mass)
"""
results = np.load('results.npy')
w1 = results[:,0]
w50 = results[:,1]
w1000 = results[:,2]

extra_b1 = np.load('extra_data/extra_b_26.npy')[-100000:]
extra_b2 = np.load('extra_data/extra_b_27.npy')[-100000:]
extra_b3 = np.load('extra_data/extra_b_28.npy')[-100000:]
extra_b4 = np.load('extra_data/extra_b_29.npy')[-100000:]
extra_b5 = np.load('extra_data/extra_b_30.npy')[-100000:]

nhits = np.concatenate((extra_b1[:,1],extra_b2[:,1],extra_b3[:,1],extra_b4[:,1],extra_b5[:,1]))
totedepecal = np.concatenate((extra_b1[:,4],extra_b2[:,4],extra_b3[:,4],extra_b4[:,4],extra_b5[:,4]))
totedepecaltight = np.concatenate((extra_b1[:,5],extra_b2[:,5],extra_b3[:,5],extra_b4[:,5],extra_b5[:,5]))
maxcell = np.concatenate((extra_b1[:,6],extra_b2[:,6],extra_b3[:,6],extra_b4[:,6],extra_b5[:,6]))
deepesthit = np.concatenate((extra_b1[:,7],extra_b2[:,7],extra_b3[:,7],extra_b4[:,7],extra_b5[:,7]))
avehit = np.concatenate((extra_b1[:,8],extra_b2[:,8],extra_b3[:,8],extra_b4[:,8],extra_b5[:,8]))
stdhit = np.concatenate((extra_b1[:,9],extra_b2[:,9],extra_b3[:,9],extra_b4[:,9],extra_b5[:,9]))
showerrms = np.concatenate((extra_b1[:,10],extra_b2[:,10],extra_b3[:,10],extra_b4[:,10],extra_b5[:,10]))
stdx = np.concatenate((extra_b1[:,11],extra_b2[:,11],extra_b3[:,11],extra_b4[:,11],extra_b5[:,11]))
stdy = np.concatenate((extra_b1[:,12],extra_b2[:,12],extra_b3[:,12],extra_b4[:,12],extra_b5[:,12]))
e_energy = np.concatenate((extra_b1[:,15],extra_b2[:,15],extra_b3[:,15],extra_b4[:,15],extra_b5[:,15]))
e_pt = np.concatenate((extra_b1[:,16],extra_b2[:,16],extra_b3[:,16],extra_b4[:,16],extra_b5[:,16]))
e_phi = np.concatenate((extra_b1[:,17],extra_b2[:,17],extra_b3[:,17],extra_b4[:,17],extra_b5[:,17]))
e_theta = np.concatenate((extra_b1[:,18],extra_b2[:,18],extra_b3[:,18],extra_b4[:,18],extra_b5[:,18]))

#nHits
plot_sb(nhits,'nHits',w1,limi=[0,200],bins=100,title='CNN, m=1 MeV')
plot_sb(nhits,'nHits',w1000,limi=[0,200],bins=100,title='CNN, m=1000 MeV')
plot_sb(nhits,'nHits',w50,limi=[0,200],bins=100,title='CNN, m=50 MeV')

#totEdepEcal
plot_sb(totedepecal,'totEdep',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
plot_sb(totedepecal,'totEdep',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(totedepecal,'totEdep',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')

#totEdepEcal
plot_sb(totedepecaltight,'totEdep tight',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
plot_sb(totedepecaltight,'totEdep tight',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(totedepecaltight,'totEdep tight',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')

#max cell Edep Ecal
plot_sb(maxcell,'max cell Edep',w1,limi=[0,1000],bins=100,title='CNN, m=1 MeV')
plot_sb(maxcell,'max cell Edep',w1000,limi=[0,1000],bins=100,title='CNN, m=1000 MeV')
plot_sb(maxcell,'max cell Edep',w50,limi=[0,1000],bins=100,title='CNN, m=50 MeV')


#deepest hit layer
plot_sb(deepesthit,'Deepest hit',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
plot_sb(deepesthit,'Deepest hit',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(deepesthit,'Deepest hit',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#average hit layer
plot_sb(avehit,'Ave hit',w1,limi=[0,30],bins=30,title='CNN, m=1 MeV')
plot_sb(avehit,'Ave hit',w1000,limi=[0,30],bins=30,title='CNN, m=1000 MeV')
plot_sb(avehit,'Ave hit',w50,limi=[0,30],bins=30,title='CNN, m=50 MeV')


#std hit layer
plot_sb(stdhit,'std hit',w1,limi=[0,16],bins=16,title='CNN, m=1 MeV')
plot_sb(stdhit,'std hit',w1000,limi=[0,16],bins=16,title='CNN, m=1000 MeV')
plot_sb(stdhit,'std hit',w50,limi=[0,16],bins=16,title='CNN, m=50 MeV')

#shower rms
plot_sb(showerrms,'shower RMS',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
plot_sb(showerrms,'shower RMS',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(showerrms,'shower RMS',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#stdX
plot_sb(stdx,'std x',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
plot_sb(stdx,'std x',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(stdx,'std x',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#stdY
plot_sb(stdy,'std y',w1,limi=[0,50],bins=50,title='CNN, m=1 MeV')
plot_sb(stdy,'std y',w1000,limi=[0,50],bins=50,title='CNN, m=1000 MeV')
plot_sb(stdy,'std y',w50,limi=[0,50],bins=50,title='CNN, m=50 MeV')


#e energy
plot_sb(e_energy,'e energy',w1,limi=[0,5000],bins=100,title='CNN, m=1 MeV')
plot_sb(e_energy,'e energy',w1000,limi=[0,5000],bins=100,title='CNN, m=1000 MeV')
plot_sb(e_energy,'e energy',w50,limi=[0,5000],bins=100,title='CNN, m=50 MeV')


#e pt
plot_sb(e_pt,'e pt',w1,limi=[0,1000],bins=100,title='CNN, m=1 MeV')
plot_sb(e_pt,'e pt',w1000,limi=[0,1000],bins=100,title='CNN, m=1000 MeV')
plot_sb(e_pt,'e pt',w50,limi=[0,1000],bins=100,title='CNN, m=50 MeV')


#e phi
plot_sb(e_phi,'e phi',w1,limi=[0,3.2],bins=40,title='CNN, m=1 MeV')
plot_sb(e_phi,'e phi',w1000,limi=[0,3.2],bins=40,title='CNN, m=1000 MeV')
plot_sb(e_phi,'e phi',w50,limi=[0,3.2],bins=40,title='CNN, m=50 MeV')


#e theta
plot_sb(e_theta,'e theta',w1,limi=[0,1.6],bins=40,title='CNN, m=1 MeV')
plot_sb(e_theta,'e theta',w1000,limi=[0,1.6],bins=40,title='CNN, m=1000 MeV')
plot_sb(e_theta,'e theta',w50,limi=[0,1.6],bins=40,title='CNN, m=50 MeV')


#empty events info
ind1 = np.where(extra_s1[:,1]==0)[0]
ind1000 = np.where(extra_s1000[:,1]==0)[0]

indb = np.where(extra_b[:,1]==0)[0]


plt.hist(extra_s1[ind1,24],np.linspace(0,500, 50))
plt.xlabel('e pt')
plt.ylabel('events')
plt.title('e pt distribution of empty events in Ecal')
plt.show()

plt.hist(extra_s1[ind1,23],np.linspace(0,1000, 50))
plt.xlabel('e energy')
plt.ylabel('events')
plt.title('e energy distribution of empty events in Ecal')
plt.show()

plt.hist(extra_s1[ind1,25],np.linspace(0,3.2, 50))
plt.xlabel('e phi')
plt.ylabel('events')
plt.title('e phi distribution of empty events in Ecal')
plt.show()

plt.hist(extra_s1[ind1,26],np.linspace(0,1.6, 50))
plt.xlabel('e theta')
plt.ylabel('events')
plt.title('e theta distribution of empty events in Ecal')
plt.show()

#performance of bdt at 99.9% background rejection for comparison

bdtb = extra_b[:,2]
bdt1 = extra_s1[:,3]
bdt5 = extra_s5[:,3]
bdt10 = extra_s10[:,3]
bdt50 = extra_s50[:,3]
bdt100 = extra_s100[:,3]
bdt500 = extra_s500[:,3]
bdt1000 = extra_s1000[:,3]

n_test = 10000
t_tab = -np.logspace(-5,0,n_test)+1
back_pass=np.zeros(n_test)
i_1_pass=np.zeros(n_test)
i_10_pass=np.zeros(n_test)
i_100_pass=np.zeros(n_test)
i_1000_pass=np.zeros(n_test)

i_5_pass=np.zeros(n_test)
i_50_pass=np.zeros(n_test)
i_500_pass=np.zeros(n_test)
for i,t in enumerate(t_tab):
    back_pass[i] = np.sum(bdtb>t)/100000 #fraction of background passing pass
    i_1_pass[i] = np.sum(bdt1>t)/100000
    i_10_pass[i] = np.sum(bdt10>t)/100000
    i_100_pass[i] = np.sum(bdt100>t)/100000
    i_1000_pass[i] = np.sum(bdt1000>t)/100000

    i_5_pass[i] = np.sum(bdt5>t)/100000
    i_50_pass[i] = np.sum(bdt50>t)/100000
    i_500_pass[i] = np.sum(bdt500>t)/100000




plt.plot(back_pass,i_1_pass,label='1 MeV')
plt.plot(back_pass,i_10_pass,label='10 MeV')
plt.plot(back_pass,i_100_pass,label='100 MeV')
plt.plot(back_pass,i_1000_pass,label='1000 MeV')
plt.plot(back_pass,i_5_pass,'--',label='5 MeV')
plt.plot(back_pass,i_50_pass,'--',label='50 MeV')
plt.plot(back_pass,i_500_pass,'--',label='500 MeV')
plt.xlabel('Fraction of background events passing cuts for BDT')
plt.ylabel('Acceptance')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.show()

#compute acceptances for 99.9% rejection for table
ind = np.argmin(np.abs(back_pass-0.001))
thresh = t_tab[ind]
print('threshold ={}'.format(thresh))
print('1 MeV: {}'.format(i_1_pass[ind]))
print('10 MeV: {}'.format(i_10_pass[ind]))
print('100 MeV: {}'.format(i_100_pass[ind]))
print('1000 MeV: {}'.format(i_1000_pass[ind]))
print('5 MeV: {}'.format(i_5_pass[ind]))
print('50 MeV: {}'.format(i_50_pass[ind]))
print('500 MeV: {}'.format(i_500_pass[ind]))
"""