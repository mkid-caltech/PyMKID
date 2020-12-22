from __future__ import division
#import os
import numpy.fft as fft
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import PyMKID_USRP_functions as PUf
#import scipy.signal as sig
#import time

c_wheel_0 = ['C0','C1','C2','C3','C4','C5','C6','C8','C9','C7']
c_wheel_1 = ['deepskyblue','sandybrown','lightgreen','lightcoral','mediumorchid','peru','lightpink','khaki','paleturquoise','silver']

def discrete_FT(data):
    return (1/len(data))*fft.fft(data)

def arclen(complex_data):
    return np.angle(complex_data*np.exp(-1j*np.angle(np.mean(complex_data,axis=0))))*np.mean(abs(complex_data),axis=0)

def CSD(data_array,time_correction,avg_num):

    chunk_len = int(0.5*len(data_array)/avg_num)
    freqs = fft.fftfreq(chunk_len,d=time_correction)
    CSD_avg = (1+1j)*np.zeros((len(data_array.T),len(data_array.T),chunk_len))
    for Nm in range(len(data_array.T)):
        for Nn in range(len(data_array.T)):
            for Am in range(avg_num):
                Am_start = int(0.5*len(data_array)+chunk_len*Am)
                Am_stop = int(0.5*len(data_array)+chunk_len*(Am+1))
                plt.plot(abs(data_array[Am_start:Am_stop,Nm]))
                plt.plot(abs(data_array[Am_start:Am_stop,Nn]))
                plt.show()
                CSD_avg[Nm,Nn] += (1/avg_num)*(time_correction*chunk_len)*np.conj(discrete_FT(data_array[Am_start:Am_stop,Nm]))*discrete_FT(data_array[Am_start:Am_stop,Nn])

    return freqs, CSD_avg

def a_estimator(s,d):
    """estimates amplitude of template given signal model s and raw data d"""
    return sum(s*d)/sum((s**2))

def noise_removal(coherent_data,removal_decimation=1):

    #Transpose data to work with templates if needed:
    coherent_data = np.transpose(coherent_data)

    coherent_data_clean= np.zeros(coherent_data.shape,dtype=np.complex64)

    #-----go through the templates and compute clean data-----
    for t in range(len(coherent_data)): #loop over all tones

        print('working on tone {}, time is {}'.format(t,datetime.now()))

        #build template with undecimated data
        off_tone_data = np.delete(coherent_data,t,axis=0) #delete tone for which template is being created #len = tones-1
        template = np.mean(off_tone_data,axis=0) #take the mean of all other tones at every sample #len = number of samples in data_noise
        template_norm = stats.zscore(template)/np.max(stats.zscore(template)) #rescales to max = 1 and mean~0

        #decimate template and data to get coeff
        ids = np.arange(len(template_norm))//removal_decimation
        template_decimated = np.bincount(ids,template_norm)/np.bincount(ids)
        coherent_data_decimated = np.bincount(ids,coherent_data[t])/np.bincount(ids)
        a1 = a_estimator(template_decimated,coherent_data_decimated) #compute the amplitude coefficient

        #clean undecimated data
        coherent_data_clean[t] = coherent_data[t] - a1*template_norm

    return np.transpose(coherent_data_clean)

def save_clean_timestreams(h5_file,data_raw,cd1_clean,cd2_clean,override=False):
    if len(cd1_clean) > 100:
        np.transpose(cd1_clean)
    if len(cd2_clean) > 100:
        np.transpose(cd2_clean)

    data_clean = cd1_clean*np.exp(1j*((cd2_clean/np.mean(abs(data_raw),axis=0))+np.angle(np.mean(data_raw,axis=0,dtype=complex))))

    with h5py.File(h5_file, 'r+') as fyle:
        if 'cleaned_data' in fyle.keys():
            print('cleaned_data already exists! If you set override=False, nothing will happen.')
            if override==True:
                print('saving clean_data to {} because override=True!'.format(h5_file))
                del fyle['cleaned_data']
                fyle['cleaned_data'] = data_clean
        else:
            print('saving clean_data to {}!'.format(h5_file))
            fyle['cleaned_data'] = data_clean

def coherence_analysis(noise_data_file,plot_file_name,extra_dec=None):
    # Plan to modify this for more than two tones eventually
    data_freqs, data_noise, time_correction = PUf.unavg_noi(noise_data_file)

    # extra_dec is necessary if significant beating in band
    if extra_dec:
        print('doing additional decimation')
        resonance_dec = sig.decimate(data_noise[:,0],extra_dec)
        tracking_dec = sig.decimate(data_noise[:,1],extra_dec)
        data_noise = np.array([resonance_dec,tracking_dec]).T
        time_correction *= extra_dec

    # Checks if the beating is too large (phase will appear as shark tooth)
    clipping = any(abs(np.mean(data_noise,axis=0)) <= np.mean(abs(data_noise-np.mean(data_noise,axis=0)),axis=0))

    # Converting to absolute and arc-length units
    coh_data_1 = abs(data_noise)
    #coh_data_2 = np.angle(data_noise*np.exp(-1j*np.angle(np.mean(data_noise,axis=0))))*np.mean(abs(data_noise),axis=0)
    coh_data_2 = arclen(data_noise)

    # Calculating cross-PSDs (for coherence and PSD comparisons)
    J_freqs, CSD_avg_1 = CSD(coh_data_1, time_correction, 33)
    J_freqs, CSD_avg_2 = CSD(coh_data_2, time_correction, 33)

    # Cleaning won't work if phase is a shark tooth
    if clipping == 0:
        # Decimate, fit noise time streams, clean undecimated data
        coh_data_1_clean = noise_removal(coh_data_1,removal_decimation=500)
        coh_data_2_clean = noise_removal(coh_data_2,removal_decimation=500)

        # Clean cross-PSDs
        J_freqs, CSD_avg_1_clean = CSD(coh_data_1_clean, time_correction, 33)
        J_freqs, CSD_avg_2_clean = CSD(coh_data_2_clean, time_correction, 33)

        # Save cleaned data back to original file
        save_clean_timestreams(noise_data_file,data_noise,coh_data_1_clean,coh_data_2_clean,override=True)

    # And a bunch of plots
    fig_0, axes_0 = plt.subplots(2,len(data_freqs)+1,sharex=True,sharey='row',figsize=(40,9))
    for Nm in range(len(data_freqs)):
        if Nm == 0:
            axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label='on-res uncleaned') # ComplexWarning: Casting complex values to real discards the imaginary part
            axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label='on-res uncleaned')
            if clipping == 0:
                axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label='on-res cleaned')
                axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label='on-res cleaned')
        elif Nm == 1:
            axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label='tracking uncleaned') # ComplexWarning: Casting complex values to real discards the imaginary part
            axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label='tracking uncleaned')
            if clipping == 0:
                axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label='tracking cleaned')
                axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label='tracking cleaned')
        axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label=str(round(data_freqs[Nm],3))+' uncleaned')
        axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label=str(round(data_freqs[Nm],3))+' uncleaned')
        if clipping == 0:
            axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label=str(round(data_freqs[Nm],3))+' cleaned')
            axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label=str(round(data_freqs[Nm],3))+' cleaned')
        axes_0[0,Nm+1].legend()
        axes_0[0,Nm+1].tick_params(labelbottom=True,labelleft=True)
        axes_0[1,Nm+1].legend()
        axes_0[1,Nm+1].tick_params(labelbottom=True,labelleft=True)
    axes_0[0,0].legend()
    axes_0[0,0].tick_params(labelbottom=True,labelleft=True)
    axes_0[1,0].legend()
    axes_0[1,0].tick_params(labelbottom=True,labelleft=True)

    fig_0.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.ylabel('absolute value PSD [(ADCu)$^2 Hz^{-1}$]',fontsize=12,labelpad=20)

    fig_0.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel('arc length PSD [(ADCu)$^2 Hz^{-1}$]',fontsize=12,labelpad=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)

    plt.savefig(plot_file_name+'_PSD.png')
    plt.close()

    fig_1, axes_1 = plt.subplots(len(data_freqs),len(data_freqs),sharex=True,sharey=True,figsize=(30,18))
    average_all_dec = 0
    #average_all_clean = 0
    for Nm in range(len(data_freqs)):
        average_Nm_dec = np.mean(np.delete((abs(CSD_avg_1[:,Nm])**2)/(np.diagonal(CSD_avg_1,axis1=0,axis2=1).T.real*CSD_avg_1[Nm,Nm].real),Nm,axis=0),axis=0)
        average_all_dec += (1/len(data_freqs))*average_Nm_dec
        axes_1[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_dec[J_freqs>0],c=c_wheel_0[Nm])
        #if clipping == 0:
        #    average_Nm_clean = np.mean(np.delete((abs(CSD_avg_1_clean[:,Nm])**2)/(np.diagonal(CSD_avg_1_clean,axis1=0,axis2=1).T.real*CSD_avg_1_clean[Nm,Nm].real),Nm,axis=0),axis=0)
        #    average_all_clean += (1/len(data_freqs))*average_Nm_clean
        #    axes_1[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_clean[J_freqs>0],'--',c=c_wheel_1[Nm])
        axes_1[Nm,Nm].tick_params(labelbottom=True,labelleft=True)
        for Nn in range(len(data_freqs)):
            if Nm < Nn:
                axes_1[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_1[Nn,Nm][J_freqs>0])**2)/(CSD_avg_1[Nn,Nn][J_freqs>0].real*CSD_avg_1[Nm,Nm][J_freqs>0].real),c='gray')
                #if clipping == 0:
                #    axes_1[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_1_clean[Nn,Nm][J_freqs>0])**2)/(CSD_avg_1_clean[Nn,Nn][J_freqs>0].real*CSD_avg_1_clean[Nm,Nm][J_freqs>0].real),'--',c='gray')
                axes_1[Nn,Nm].legend(handles=[mpatches.Patch(color=c_wheel_0[Nn]),mpatches.Patch(color=c_wheel_0[Nm])])
                axes_1[Nn,Nm].tick_params(labelbottom=True,labelleft=True)
            elif Nm != Nn:
                axes_1[Nn,Nm].axis('off')
    axes_1[0,len(data_freqs)-1].axis('on')
    axes_1[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_dec[J_freqs>0],c='k')
    #if clipping == 0:
    #    axes_1[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_clean[J_freqs>0],'--',c='k')
    axes_1[0,len(data_freqs)-1].tick_params(labelbottom=True,labelleft=True)

    fig_1.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)
    plt.ylabel('absolute value coherence',fontsize=25,labelpad=20)

    plt.savefig(plot_file_name+'_coh_a.png')
    plt.close()

    fig_2, axes_2 = plt.subplots(len(data_freqs),len(data_freqs),sharex=True,sharey=True,figsize=(30,18))
    average_all_dec = 0
    #average_all_clean = 0
    for Nm in range(len(data_freqs)):
        average_Nm_dec = np.mean(np.delete((abs(CSD_avg_2[:,Nm])**2)/(np.diagonal(CSD_avg_2,axis1=0,axis2=1).T.real*CSD_avg_2[Nm,Nm].real),Nm,axis=0),axis=0)
        average_all_dec += (1/len(data_freqs))*average_Nm_dec
        axes_2[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_dec[J_freqs>0],c=c_wheel_0[Nm])
        #if clipping == 0:
        #    average_Nm_clean = np.mean(np.delete((abs(CSD_avg_2_clean[:,Nm])**2)/(np.diagonal(CSD_avg_2_clean,axis1=0,axis2=1).T.real*CSD_avg_2_clean[Nm,Nm].real),Nm,axis=0),axis=0)
        #    average_all_clean += (1/len(data_freqs))*average_Nm_clean
        #    axes_2[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_clean[J_freqs>0],'--',c=c_wheel_1[Nm])
        axes_2[Nm,Nm].tick_params(labelbottom=True,labelleft=True)
        for Nn in range(len(data_freqs)):
            if Nm < Nn:
                axes_2[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_2[Nn,Nm][J_freqs>0])**2)/(CSD_avg_2[Nn,Nn][J_freqs>0].real*CSD_avg_2[Nm,Nm][J_freqs>0].real),c='gray')
                #if clipping == 0:
                #    axes_2[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_2_clean[Nn,Nm][J_freqs>0])**2)/(CSD_avg_2_clean[Nn,Nn][J_freqs>0].real*CSD_avg_2_clean[Nm,Nm][J_freqs>0].real),'--',c='gray')
                axes_2[Nn,Nm].legend(handles=[mpatches.Patch(color=c_wheel_0[Nn]),mpatches.Patch(color=c_wheel_0[Nm])])
                axes_2[Nn,Nm].tick_params(labelbottom=True,labelleft=True)
            elif Nm != Nn:
                axes_2[Nn,Nm].axis('off')
    axes_2[0,len(data_freqs)-1].axis('on')
    axes_2[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_dec[J_freqs>0],c='k')
    #if clipping == 0:
    #    axes_2[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_clean[J_freqs>0],'--',c='k')
    axes_2[0,len(data_freqs)-1].tick_params(labelbottom=True,labelleft=True)

    fig_2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)
    plt.ylabel('arc length coherence',fontsize=25,labelpad=20)

    plt.savefig(plot_file_name+'_coh_b.png')
    plt.close()
