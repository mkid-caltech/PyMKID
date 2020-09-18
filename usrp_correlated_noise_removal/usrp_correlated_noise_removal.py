#### Alvaro Loya Villalapndo - aloyavil@caltech.edu #####
#### Code to remove correrlated 1/f noise

from __future__ import division
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy import stats
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 #for plotting large amounts of data

path_dec_plots = "/Users/alvaro/Desktop/Caltech/group/analysis/usrp_noise_correlation/dec_vs_undec" #change to your working directory
path_clean_vs_dirty_plots = "/Users/alvaro/Desktop/Caltech/group/analysis/usrp_noise_correlation/cvd" #change to your working directory

    
def unavg_noi(filename,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3] #picks out date and last number of file name separated with '_'

    with h5py.File(filename, "r") as fyle:
        raw_noise = get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

        eff_rate = rate/decimation
        time_correction = 1/eff_rate # s?
        
    if verbose:
        print("\n\nData taken "+str(Dt_tm))
        print("Reported LO is "+str(LO*1e-6)+" MHz")
        print("Reported rate is %f MHz"%(rate/1e6))
        print("Reported decimation is %d"%(decimation))
        print("\tEffective rate is %f kHz"%(eff_rate/1e3))
        print("Reported amplitudes are "+str(amplitude))
        print("\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm") #amplitude to power conversion
        print("Tones are "+str(search_freqs*1e-6)+" MHz")

    return search_freqs*1e-6, raw_noise, time_correction
        
def get_raw(openfile):
    raw_data = np.array(openfile["raw_data0/A_RX2/data"])
    return np.transpose(raw_data) #transpose of the raw data

def calc_mag_phase(data_noise):
    "calculated the amplitude and arc length of the data"
    coh_data_1 = abs(data_noise) #magnitude
    coh_data_2 = np.angle(data_noise*np.exp(-1j*np.angle(np.mean(data_noise,axis=0))))*np.mean(abs(data_noise),axis=0) #arclength = delta_angle * radi

    return (coh_data_1,coh_data_2)

def reduced_data(data,start,end):
    "returns reduced amount of data from point start to point end"
    return data[start:end]

def further_decimation(decimation,coh_data_1,coh_data_2,time_correction_undecimated,plot=False,data='test_name'):

    "returns decimated amplitude and phase decimated data and a decimated time correction. Also stores plots of decimated vs undecimated data if plot=True"
    
    if plot==True:
        try:
            os.makedirs(path_dec_plots)
        except OSError:
            print ("Creation of the directory %s failed, it may already exist" % path_dec_plots)
        else:
            print ("Successfully created the directory %s" % path_dec_plots)
    
    if len(coh_data_1) > 100:
        coh_data_1=np.transpose(coh_data_1)
        coh_data_2=np.transpose(coh_data_2)
        

    #create empty arrays
    coh_data_1_decimated= np.zeros((len(coh_data_1),int(len(coh_data_1[0])/decimation)))
    coh_data_2_decimated= np.zeros((len(coh_data_1),int(len(coh_data_2[0])/decimation)))

    if decimation > 1:
        time_correction_decimated = time_correction_undecimated*decimation
        
        for i in range(len(coh_data_1)):
            ids = np.arange(len(coh_data_1[i]))//decimation
            coh_data_1_decimated[i] = np.bincount(ids,coh_data_1[i])/np.bincount(ids)

            if plot==True:
                x1=np.arange(0,len(coh_data_1[i]),1)
                x2 = np.arange(0,len(coh_data_1_decimated[i])*decimation,decimation)

                plt.figure()
                plt.title('undecimated(b) vs decimated(r) amplitude - tone{}'.format(i))
                plt.plot(x1,20*np.log10(abs(coh_data_1[i])),label='undecimated',color='b')
                plt.plot(x2,20*np.log10(abs(coh_data_1_decimated[i])),label='decimated',color='r')
                plt.xlabel('time sample')
                plt.ylabel('S21 (dB)')
                plt.xlim(1000,np.max(x1))
                plt.ylim(np.min(20*np.log10(abs(coh_data_1[i][1000:]))),np.max(20*np.log10(abs(coh_data_1[i][1000:]))))
                plt.savefig('./dec_vs_undec/'+data[:-3]+'_undecimated_vs_decimated_amplitude_decimation_'+str(decimation)+'_tone{}.png'.format(i))
                plt.close()
                
                plt.figure()
                plt.title('decimated amplitude - tone{}'.format(i))
                plt.plot(x2,20*np.log10(abs(coh_data_1_decimated[i])),label='decimated',color='r')
                plt.xlabel('time sample')
                plt.ylabel('S21 (dB)')
                plt.xlim(1000,np.max(x2))
                plt.ylim(np.min(20*np.log10(abs(coh_data_1_decimated[i][1000:]))),np.max(20*np.log10(abs(coh_data_1_decimated[i][1000:]))))
                plt.savefig('./dec_vs_undec/'+data[:-3]+'_decimated_amplitude_decimation_'+str(decimation)+'_tone{}.png'.format(i))
                plt.close()

            ids = np.arange(len(coh_data_2[i]))//decimation
            coh_data_2_decimated[i] = np.bincount(ids,coh_data_2[i])/np.bincount(ids)

            if plot==True:
                x1=np.arange(0,len(coh_data_2[i]),1)
                x2 = np.arange(0,len(coh_data_2_decimated[i])*decimation,decimation)

                plt.figure()
                plt.title('undecimated(b) vs decimated(r) phase - tone{}'.format(i))
                plt.plot(x1,coh_data_2[i],label='undecimated',color='b')
                plt.plot(x2,coh_data_2_decimated[i],label='decimated',color='r')
                plt.xlabel('time sample')
                plt.xlim(1000,np.max(x1))
                plt.ylim(np.min(coh_data_2[i][1000:]),np.max(coh_data_2[i][1000:]))
                plt.savefig('./dec_vs_undec/'+data[:-3]+'_undecimated_vs_decimated_phase_decimation_'+str(decimation)+'_tone{}.png'.format(i))
                plt.close()
                
                plt.figure()
                plt.title('decimated phase - tone{}'.format(i))
                plt.plot(x2,coh_data_2_decimated[i],label='decimated',color='r')
                plt.xlabel('time sample')
                plt.xlim(1000,np.max(x2))
                plt.ylim(np.min(coh_data_2_decimated[i][1000:]),np.max(coh_data_2_decimated[i][1000:]))
                plt.savefig('./dec_vs_undec/'+data[:-3]+'_decimated_phase_decimation_'+str(decimation)+'_tone{}.png'.format(i))
                plt.close()

    coh_data_1_decimated=np.transpose(coh_data_1_decimated)
    coh_data_2_decimated=np.transpose(coh_data_2_decimated)

    return (coh_data_1_decimated,coh_data_2_decimated,time_correction_decimated)
    
def compute_spectrum(coh_data_1,coh_data_1_decimated,time_correction_undecimated,time_correction_decimated,decimated_spectrum=False,avg=33):
                 
    "Returns frequency spectrum for PSD. decimated_spectrum determines if the spectrum will be calculated with decimated or undecimated data"

    avg_num = avg

    if len(coh_data_1_decimated) < 100:
        coh_data_1_decimated=np.transpose(coh_data_1_decimated)
        
    if len(coh_data_1) < 100:
        coh_data_1=np.transpose(coh_data_1)
        

    if decimated_spectrum==True:
        chunk_len = int(0.5*len(coh_data_1_decimated)/avg_num)
    else:
        chunk_len = int(0.5*len(coh_data_1)/avg_num)

    if decimated_spectrum==True:
        J_freqs = fft.fftfreq(chunk_len,d=time_correction_decimated) #(window_length=151515, sample spacing = 0.2us)
    else:
        J_freqs = fft.fftfreq(chunk_len,d=time_correction_undecimated)

    return J_freqs

def a_estimator(s,d):
    """estimates amplitude of template given signal model s and raw data d"""
    t = 0
    b = 0
    counter = 0
    for i in range(len(s)):
        t += s[i]*d[i]
        b += s[i]**2
    return t/b

def StandardizeData(data):
    """standardizes data to have a mean of 0 and a standard deviation of 1"""
    return stats.zscore(data)
    
def noise_removal(coh_data_1,coh_data_2, coh_data_1_decimated,coh_data_2_decimated,decimation,data_freqs,decimated_data=True,ommitted_vals=[],plot=False,data='test_name'):
    
    if plot==True:
        try:
            os.makedirs(path_clean_vs_dirty_plots)
        except OSError:
            print ("Creation of the directory %s failed, it may already exist" % path_clean_vs_dirty_plots)
        else:
            print ("Successfully created the directory %s" % path_clean_vs_dirty_plots)
            
    #Transpose data to work with templates if needed:
    if len(coh_data_1) > 100: #or any other value of tones
        coh_data_1 = np.transpose(coh_data_1)
        coh_data_2 = np.transpose(coh_data_2)
    if len(coh_data_1_decimated) > 100: #or any other value of tones
        coh_data_1_decimated = np.transpose(coh_data_1_decimated)
        coh_data_2_decimated = np.transpose(coh_data_2_decimated)

    if decimated_data == True:
        #-----create empty arrays-----
        coh_data_1_clean= np.zeros((len(coh_data_1_decimated)-len(ommitted_vals),len(coh_data_1_decimated[0])),dtype=np.complex64)
        coeff1 = np.zeros((len(coh_data_1_decimated)-len(ommitted_vals)))
        coh_data_2_clean= np.zeros((len(coh_data_2_decimated)-len(ommitted_vals),len(coh_data_2_decimated[0])),dtype=np.complex64)
        coeff2 = np.zeros((len(coh_data_2_decimated)-len(ommitted_vals)))

        #----for storing only tones used
        coh_data_1_filtered = np.zeros((len(coh_data_1_decimated)-len(ommitted_vals),len(coh_data_1_decimated[0])),dtype=np.complex64)
        coh_data_2_filtered = np.zeros((len(coh_data_2_decimated)-len(ommitted_vals),len(coh_data_2_decimated[0])),dtype=np.complex64)

    elif decimated_data == False:
        #-----create empty arrays-----
        coh_data_1_clean= np.zeros((len(coh_data_1)-len(ommitted_vals),len(coh_data_1[0])),dtype=np.complex64)
        coeff1 = np.zeros((len(coh_data_1)-len(ommitted_vals)))
        coh_data_2_clean= np.zeros((len(coh_data_2)-len(ommitted_vals),len(coh_data_2[0])),dtype=np.complex64)
        coeff2 = np.zeros((len(coh_data_2)-len(ommitted_vals)))

        #----for storing only tones used
        coh_data_1_filtered = np.zeros((len(coh_data_1)-len(ommitted_vals),len(coh_data_1[0])),dtype=np.complex64)
        coh_data_2_filtered = np.zeros((len(coh_data_2)-len(ommitted_vals),len(coh_data_2[0])),dtype=np.complex64)

    #---for storing the frequencies considered----
    data_freqs_filtered = np.zeros((len(data_freqs)-len(ommitted_vals)))

    c = 0 #counter to keep track of tones included

    #-----go through the templates and compute clean data-----
    for t in range(len(coh_data_1)): #loop over all tones
        if t in ommitted_vals: #call out which templates will not be cleaned nor used for templates
            print('not cleaning tone {} nor using it for building templates'.format(t))
            continue

        #store good data freqs / tones
        data_freqs_filtered[c] = data_freqs[t]

        print('working on tone {}, time is {}'.format(t,datetime.now()))

        #----amplitude----
        print('working on magnitude', datetime.now())

        if decimated_data==True:
            label = 'decimated_data' #for saving plots
            #build template with decimated data
            ommitted_vals.append(t) #add current tone to list of tones not considered in template
            tones1_dec = np.delete(coh_data_1_decimated,ommitted_vals,axis=0) #delete tone for which template is being created #len = tones-1
            ommitted_vals.pop(-1) #delete current tone from excluded values
            s1_dec = np.mean(tones1_dec,axis=0) #take the mean of all other tones at every sample #len = number of samples in data_noise
            s1_stand_dec = StandardizeData(s1_dec) #standardize (mean 0 and std = 1)
            s1_norm_dec = s1_stand_dec/np.max(s1_stand_dec) #rescales to max = 1 while maintaining mean~0
            a1 = a_estimator(s1_norm_dec,coh_data_1_decimated[t]) #compute the amplitude coefficient
            coeff1[c] = a1 #store amplitude coefficient
            
            #compute clean undecimated data
            coh_data_1_clean[c] = coh_data_1_decimated[t] - a1*s1_norm_dec
            coh_data_1_filtered[c] = coh_data_1_decimated[t]
            
        if decimated_data==False: #this is the best cleanup
            label = 'undecimated_data' #for saving plots
            #build template with undecimated data
            ommitted_vals.append(t) #add current tone to list of tones not considered in template
            tones1 = np.delete(coh_data_1,ommitted_vals,axis=0) #delete tone for which template is being created #len = tones-1
            ommitted_vals.pop(-1) #delete current tone from excluded values
            s1 = np.mean(tones1,axis=0) #take the mean of all other tones at every sample #len = number of samples in data_noise
            s1_stand = StandardizeData(s1) #standardize (mean 0 and std = 1)
            s1_norm = s1_stand/np.max(s1_stand) #rescales to max = 1 while maintaining mean~0
            
            #decimate template to get coeff for undecimated data
            ids = np.arange(len(s1_norm))//decimation
            s1_scaled = np.bincount(ids,s1_norm)/np.bincount(ids)
            a1 = a_estimator(s1_scaled,coh_data_1_decimated[t]) #compute the amplitude coefficient
            coeff1[c] = a1 #store amplitude coefficient
            
            #compute clean undecimated data
            coh_data_1_clean[c] = coh_data_1[t] - a1*s1_norm
            coh_data_1_filtered[c] = coh_data_1[t]


        #plots
        
        if plot==True:
            plt.figure()
            plt.title('clean (b) vs dirty (r) amplitude - tone{}'.format(t))
            plt.plot(np.arange(0,len(coh_data_1_filtered[c]),1),coh_data_1_filtered[c],color='r',label='dirty')
            plt.plot(np.arange(0,len(coh_data_1_clean[c]),1),coh_data_1_clean[c],color='b',label='clean',alpha=0.1)
            plt.xlim(1000,np.max(np.arange(0,len(coh_data_1_filtered[c]),1)))
            plt.ylim(np.min(coh_data_1_filtered[c][1000:]),np.max(coh_data_1_filtered[c][1000:]))
            plt.savefig('./cvd/'+data[:-3]+'_'+label+'_clean_vs_dirty_amplitude_tone{}.png'.format(t))
            plt.close()
     
         #----arclength----
        print('working on phase', datetime.now())
        
        if decimated_data==True:
            label = 'decimated_data' #for saving plots
            #build template with decimated data
            ommitted_vals.append(t) #add current tone to list of tones not considered in template
            tones2_dec = np.delete(coh_data_2_decimated,ommitted_vals,axis=0) #delete tone for which template is being created #len = tones-1
            ommitted_vals.pop(-1) #delete current tone from excluded values
            s2_dec = np.mean(tones1_dec,axis=0) #take the mean of all other tones at every sample #len = number of samples in data_noise
            s2_stand_dec = StandardizeData(s2_dec) #standardize (mean 0 and std = 1)
            s2_norm_dec = s2_stand_dec/np.max(s2_stand_dec) #rescales to max = 1 while maintaining mean~0
            a2 = a_estimator(s2_norm_dec,coh_data_2_decimated[t]) #compute the amplitude coefficient
            coeff2[c] = a2 #store amplitude coefficient
            
            #compute clean undecimated data
            coh_data_2_clean[c] = coh_data_2_decimated[t] - a2*s2_norm_dec
            coh_data_2_filtered[c] = coh_data_2_decimated[t]
            
        if decimated_data==False: #this is the best cleanup
            label = 'undecimated_data' #for saving plots
            #build template with undecimated data
            ommitted_vals.append(t) #add current tone to list of tones not considered in template
            tones2 = np.delete(coh_data_2,ommitted_vals,axis=0) #delete tone for which template is being created #len = tones-1
            ommitted_vals.pop(-1) #delete current tone from excluded values
            s2 = np.mean(tones2,axis=0) #take the mean of all other tones at every sample #len = number of samples in data_noise
            s2_stand = StandardizeData(s2) #standardize (mean 0 and std = 1)
            s2_norm = s2_stand/np.max(s2_stand) #rescales to max = 1 while maintaining mean~0
            
            #decimate template to get coeff for undecimated data
            ids = np.arange(len(s2_norm))//decimation
            s2_scaled = np.bincount(ids,s2_norm)/np.bincount(ids)
            a2 = a_estimator(s2_scaled,coh_data_2_decimated[t]) #compute the amplitude coefficient
            coeff2[c] = a2 #store amplitude coefficient
            
            #compute clean undecimated data
            coh_data_2_clean[c] = coh_data_2[t] - a2*s2_norm
            coh_data_2_filtered[c] = coh_data_2[t]
        
        if plot==True:
            plt.figure()
            plt.title('clean (b) vs dirty (r) phase - tone{}'.format(t))
            plt.plot(np.arange(0,len(coh_data_2_filtered[c]),1),coh_data_2_filtered[c],color='r',label='dirty')
            plt.plot(np.arange(0,len(coh_data_2_clean[c]),1),coh_data_2_clean[c],color='b',label='clean',alpha=0.1)
            plt.xlim(1000,np.max(np.arange(0,len(coh_data_2_filtered[c]),1)))
            plt.ylim(np.min(coh_data_2_filtered[c][1000:]),np.max(coh_data_2_filtered[c][1000:]))
            plt.savefig('./cvd/'+data[:-3]+'_'+label+'_clean_vs_dirty_phase_tone{}.png'.format(t))
            plt.close()

        c +=1
        
    return (data_freqs_filtered,coeff1,coeff2,np.transpose(coh_data_1_filtered),np.transpose(coh_data_2_filtered),np.transpose(coh_data_1_clean),np.transpose(coh_data_2_clean))


#PSD calculation and plotting below

def discrete_FT(data):
    return (1/len(data))*fft.fft(data)

c_wheel = ['C0','C1','C2','C3','C4','C5','C6','C8','C9','C7','C8','C9']
data_type_name_1 = ['absolute value','fractional absolute value',r'$\delta \frac{1}{Q_i}$',r'$n_{qp}(\kappa_1)$']
data_type_name_2 = ['arc length', 'angle',r'$\frac{\delta f}{f}$',r'$n_{qp}(\kappa_2)$']
data_type_unit_1 = [' ',' ',' ',r'$\mu m^{-3}$']
data_type_unit_2 = [' ','rad',' ',r'$\mu m^{-3}$']
data_type = 0
data = 'test_data'

def psd_calc(data_freqs,coh_data_1,coh_data_2,time_correction,avg_num=33,print_progress=True):

    "returns PSDs of coh_data_1 and coh_data_2"
    
    chunk_len = int(0.5*len(coh_data_1)/avg_num)
    
    for Nm in range(len(data_freqs)):
        for Nn in range(len(data_freqs)):
            if print_progress==True:
                print(Nm, Nn)
            if Nm == 0 and Nn == 0:
                PSD_avg_1 = (1+1j)*np.zeros((len(data_freqs),len(data_freqs),chunk_len))
                PSD_avg_2 = (1+1j)*np.zeros((len(data_freqs),len(data_freqs),chunk_len))
            for Am in range(avg_num):
                Am_start = int(0.5*len(coh_data_1)+chunk_len*Am)
                Am_stop = int(0.5*len(coh_data_1)+chunk_len*(Am+1))
                PSD_avg_1[Nm,Nn] += (1/avg_num)*(time_correction*chunk_len)*np.conj(discrete_FT(coh_data_1[Am_start:Am_stop,Nm]))*discrete_FT(coh_data_1[Am_start:Am_stop,Nn])
                PSD_avg_2[Nm,Nn] += (1/avg_num)*(time_correction*chunk_len)*np.conj(discrete_FT(coh_data_2[Am_start:Am_stop,Nm]))*discrete_FT(coh_data_2[Am_start:Am_stop,Nn])

    return (PSD_avg_1,PSD_avg_2)
    
def psd_plot(data_freqs,J_freqs,PSD_avg_1,PSD_avg_2,PSD_avg_1_clean,PSD_avg_2_clean,data='test_name'):
    "plots the clean and dirty PDSs for each tone"

    fig_0, axes_0 = plt.subplots(2,len(data_freqs)+1,sharex=True,sharey='row',figsize=(40,9))
    for Nm in range(len(data_freqs)):
        if len(J_freqs) == 0:
            print('nothin in J_freqs, check dimension of data_noise (10000000 not 8!)')
            break
        axes_0[0,0].loglog(J_freqs[J_freqs>0],PSD_avg_1[Nm,Nm][J_freqs>0].real,'--',c=c_wheel[Nm]) # ComplexWarning: Casting complex values to real discards the imaginary part
        axes_0[1,0].loglog(J_freqs[J_freqs>0],PSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel[Nm])
        axes_0[0,0].loglog(J_freqs[J_freqs>0],PSD_avg_1_clean[Nm,Nm][J_freqs>0].real,'-',c=c_wheel[Nm+1]) # ComplexWarning: Casting complex values to real discards the imaginary part
        axes_0[1,0].loglog(J_freqs[J_freqs>0],PSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'-',c=c_wheel[Nm+1])
        #dirty data
        axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],PSD_avg_1[Nm,Nm][J_freqs>0].real,'--',c=c_wheel[Nm],label=str(round(data_freqs[Nm],3))+'-dirty')
        axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],PSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel[Nm],label=str(round(data_freqs[Nm],3))+'-dirty')
        #clean data
        axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],PSD_avg_1_clean[Nm,Nm][J_freqs>0].real,'-',c=c_wheel[Nm+1],label=str(round(data_freqs[Nm],3))+'-clean')
        axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],PSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'-',c=c_wheel[Nm+1],label=str(round(data_freqs[Nm],3))+'-clean')
        
        
        axes_0[0,Nm+1].legend()
        axes_0[1,Nm+1].legend()

    fig_0.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.ylabel(data_type_name_1[data_type]+' PSD [('+data_type_unit_1[data_type]+')$^2 Hz^{-1}$]',fontsize=12,labelpad=20)

    fig_0.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(data_type_name_2[data_type]+' PSD [('+data_type_unit_2[data_type]+')$^2 Hz^{-1}$]',fontsize=12,labelpad=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)

    plt.savefig('./psd_clean_vs_dirty_'+data[:-3]+'_PSD.png')
    plt.close()

def save_clean_timestreams(h5_file,data_raw,cd1,cd2,cd1_clean,cd2_clean,coeff1,coeff2,override=True):
    if len(cd1) > 100:
        np.transpose(cd1)
    if len(cd2) > 100:
        np.transpose(cd2)
    if len(cd1_clean) > 100:
        np.transpose(cd1_clean)
    if len(cd2_clean) > 100:
        np.transpose(cd2_clean)
        
    data_dirty = cd1*np.exp(1j*((cd2/np.mean(abs(data_raw),axis=0))+np.angle(np.mean(data_raw,axis=0))))
    data_clean = cd1_clean*np.exp(1j*((cd2_clean/np.mean(abs(data_raw),axis=0))+np.angle(np.mean(data_raw,axis=0))))
    
    with h5py.File(h5_file, 'r+') as fyle:
        if 'cleaned_data' in fyle.keys():
            print('cleaned_data already exists! If you set override=False, nothing will happen.')
            if override==True:
                print('saving clean_data to {}!'.format(h5_file))
                del fyle['cleaned_data']
                fyle['cleaned_data'] = data_clean
