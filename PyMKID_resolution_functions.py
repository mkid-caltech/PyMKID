from __future__ import division
#import os
import numpy.fft as fft
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import h5py
#import Taylors_USRP_functions as tuf
#import scipy.signal as sig
#import time

def discrete_FT(data):
    return (1/len(data))*fft.fft(data)

def CSD(data_array,time_correction,avg_num):

    chunk_len = int(0.5*len(data_array)/avg_num)
    freqs = fft.fftfreq(chunk_len,d=time_correction)
    CSD_avg = (1+1j)*np.zeros((len(data_array.T),len(data_array.T),chunk_len))
    for Nm in range(len(data_array.T)):
        for Nn in range(len(data_array.T)):
            for Am in range(avg_num):
                Am_start = int(0.5*len(data_array)+chunk_len*Am)
                Am_stop = int(0.5*len(data_array)+chunk_len*(Am+1))
                CSD_avg[Nm,Nn] += (1/avg_num)*(time_correction*chunk_len)*np.conj(discrete_FT(data_array[Am_start:Am_stop,Nm]))*discrete_FT(data_array[Am_start:Am_stop,Nn])

    return freqs, CSD_avg
