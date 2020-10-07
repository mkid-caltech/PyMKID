from __future__ import division
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
import h5py
import fitres as fitres
import matplotlib.pyplot as plt

#print fyle["raw_data0/A_RX2"].attrs.keys()
#print fyle["raw_data0/A_RX2"].attrs.values()

def template(filename,time_threshold=20e-3,ythreshold=0.01,left_time=2e-3,right_time=28e-3,pulse_width=20e-6,osmond=False,period=None):
    trigNum=0
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_noise = get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    eff_rate = rate/decimation

    time_correction = 1/eff_rate
    left_len = int(left_time*eff_rate)
    right_len = int(right_time*eff_rate)
    pulse_len = int(pulse_width*eff_rate)
    total_len = left_len + right_len
    xthreshold = int(time_threshold*eff_rate)
    all_channels = np.mean(abs(raw_noise-np.mean(raw_noise,axis=0)),axis=1)

    temp_time = np.array(range(left_len+right_len))*time_correction

    if True:
        plt.figure(1)
        plt.plot(np.array(range(len(all_channels)))*time_correction,all_channels, label="stuff")
        plt.axhline(y=ythreshold,ls='--',c='gray')
        plt.axvline(x=max(xthreshold,left_len)*time_correction,ls='--',c='gray')
        plt.axvline(x=(len(raw_noise)-right_len)*time_correction,ls='--',c='gray')
        plt.xlabel("time [s]")
        plt.legend()
        plt.show()

    if period != None:
        time_array = np.array(range(len(all_channels)))*time_correction
        temp_time = time_array[0:int(period*eff_rate)]
        trigNum = int(max(time_array)/period)
        for xx in range(trigNum):
            if xx==0:
                temp_array = raw_noise[0:int(period*eff_rate)]
            else:
                temp_array += raw_noise[xx*int(period*eff_rate):(xx+1)*int(period*eff_rate)]
    elif osmond:
        window_size = 10000
        xx = xthreshold + int(window_size/2) - 1
        window_mean = np.mean(all_channels[xthreshold:xthreshold+window_size])
        window_var = np.var(all_channels[xthreshold:xthreshold+window_size])
        actual_trigNum = 0
        while actual_trigNum == 0:
            if all_channels[xx] > window_mean + 6*np.sqrt(window_var):
                actual_trigNum = 1
                temp_array = raw_noise[xx-left_len:xx+right_len]
                # plt.plot(np.array(range(len(all_channels)))*time_correction,all_channels, label="stuff")
                # plt.figure()
                # plt.plot(np.array(range(xx-left_len,xx+right_len))*time_correction,all_channels[xx-left_len:xx+right_len])
                # plt.show()
                print 'found first trigger'
                break
            first = all_channels[xx-int(window_size/2)+1]
            last = all_channels[xx+int(window_size/2)+1]
            window_var -= (first-window_mean)**2/window_size
            window_var += (last-window_mean)**2/window_size
            window_mean -= first/window_size
            window_mean += last/window_size
            xx += 1
            if xx == len(all_channels)/10:
                print 'no  first trigger found'


        xx += int(100e-3*eff_rate)
        while actual_trigNum != 0 and xx < len(all_channels) - right_len:
            window = all_channels[xx-int(window_size/2):xx+int(window_size/2)]
            window_mean = np.mean(window)
            window_std = np.std(window)
            # plt.plot(range(len(window)),window)
            # plt.show()
            peaks = np.argwhere(window > window_mean + 6*window_std)
            # print peaks[0]
            if len(peaks)<2:
                xx += int(100e-3*eff_rate)
                print 'not enough peaks found'
                continue
            xx = peaks[0][0] + xx - int(window_size/2)
            if actual_trigNum > 9: # only want the second second, according to Taylor
                trigNum += 1
                actual_trigNum += 1
                temp_array += raw_noise[xx-left_len:xx+right_len]
            else:
                actual_trigNum += 1
            xx += int(100e-3*eff_rate)

    else:
        for xx in range(len(all_channels)):
            if xx > max(xthreshold,left_len) and xx < (len(all_channels)-right_len):
                if all_channels[xx] > ythreshold:
                    trigNum += 1
                    if trigNum == 1:
                        temp_array = raw_noise[xx-left_len:xx+right_len]
                    else:
                        temp_array += raw_noise[xx-left_len:xx+right_len]
                    all_channels[xx-left_len:xx+right_len] = 0


    print "Found %d triggering events"%(trigNum)

    return trigNum, temp_array, temp_time, search_freqs

def vna_file_fit(filename,pickedres,show=False,save=False):
    pickedres = np.array(pickedres)
    VNA_f, VNA_z = read_vna(filename, decimation=1)
    VNA_f = VNA_f*1e-3
    frs = np.zeros(len(pickedres))
    Qrs = np.zeros(len(pickedres))
    for MKIDnum in range(len(pickedres)):
        MKID_index = np.argmin(abs(VNA_f-pickedres[MKIDnum]))
        index_range = 5000
        MKID_f = VNA_f[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]
        MKID_z = VNA_z[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]
        frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau, Qc = fitres.finefit(MKID_f, MKID_z, pickedres[MKIDnum])

        if show:
            fit_z = fitres.resfunc3(MKID_f, frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau)
            #MKID_z_corrected = 1-((1-MKID_z/(a*np.exp(-2j*np.pi*(MKID_f-frs[MKIDnum])*tau)))*(np.cos(phi)/np.exp(1j*phi)))
            #fit_z_corrected = 1-(Qrs[MKIDnum]/Qc)/(1+2j*Qrs[MKIDnum]*(MKID_f-frs[MKIDnum])/frs[MKIDnum])
            plt.plot(MKID_f,20*np.log10(abs(MKID_z)))
            plt.plot(MKID_f,20*np.log10(abs(fit_z)))
            plt.show()
        if save:
            fit_z = fitres.resfunc3(MKID_f, frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau)
            #MKID_z_corrected = 1-((1-MKID_z/(a*np.exp(-2j*np.pi*(MKID_f-frs[MKIDnum])*tau)))*(np.cos(phi)/np.exp(1j*phi)))
            #fit_z_corrected = 1-(Qrs[MKIDnum]/Qc)/(1+2j*Qrs[MKIDnum]*(MKID_f-frs[MKIDnum])/frs[MKIDnum])
            plt.plot(MKID_f,20*np.log10(abs(MKID_z)))
            plt.plot(MKID_f,20*np.log10(abs(fit_z)))
            plt.savefig(filename[:-3]+'_res'+str(MKIDnum)+'.png')
            plt.close()

    return frs, Qrs

def get_raw(openfile):
    raw_data = np.array(openfile["raw_data0/A_RX2/data"])
    return np.transpose(raw_data)

def clean_noi(file):
    with h5py.File(file, "r") as fyle:
        cleaned_data = np.array(fyle["cleaned_data"])
    return cleaned_data

def read_vna(filename, decimation=1,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_VNA = get_raw(fyle)
        amplitude = fyle["raw_data0/A_RX2"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        f0 = fyle["raw_data0/A_RX2"].attrs.get('freq')[0]
        f1 = fyle["raw_data0/A_RX2"].attrs.get('chirp_f')[0]
        delay = (fyle["raw_data0/A_RX2"].attrs.get('delay')-1)*1e9 # ns

    eff_rate = rate/decimation

    if verbose:
        print "\n\nData taken "+str(Dt_tm)
        print "Reported LO is "+str(LO*1e-6)+" MHz"
        print "Reported rate is %f MHz"%(rate/1e6)
        print "Entered decimation is %d"%(decimation)
        print "\tEffective rate is %f kHz"%(eff_rate/1e3)
        print "Reported amplitudes are "+str(amplitude)
        print "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm"

    raw_f = np.linspace(LO+f0,LO+f1,len(raw_VNA[:,0]))*1e-6

    return raw_f, raw_VNA[:,0]

def avg_noi(filename,time_threshold=0.05,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_noise = get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    if verbose:
        print "\n\nData taken "+str(Dt_tm)
        print "Reported LO is "+str(LO*1e-6)+" MHz"
        print "Reported rate is %f MHz"%(rate/1e6)
        print "Reported decimation is %d"%(decimation)
        print "\tEffective rate is %f kHz"%(eff_rate/1e3)
        print "Reported amplitudes are "+str(amplitude)
        print "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm"
        print "Tones are "+str(search_freqs*1e-6)+" MHz"

    eff_rate = rate/decimation
    time_correction = 1/eff_rate
    time_array = time_correction*np.arange(0,len(raw_noise))
    array_mean = np.mean(raw_noise[time_array>time_threshold], axis=0)

    return search_freqs*1e-6, array_mean

def unavg_noi(filename,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_noise = get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    if verbose:
        print "\n\nData taken "+str(Dt_tm)
        print "Reported LO is "+str(LO*1e-6)+" MHz"
        print "Reported rate is %f MHz"%(rate/1e6)
        print "Reported decimation is %d"%(decimation)
        print "\tEffective rate is %f kHz"%(eff_rate/1e3)
        print "Reported amplitudes are "+str(amplitude)
        print "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm"
        print "Tones are "+str(search_freqs*1e-6)+" MHz"

    eff_rate = rate/decimation
    time_correction = 1/eff_rate # s?

    return search_freqs*1e-6, raw_noise, time_correction

def avg_VNA(filename, decimation=1, f0=None, f1=None, targets=None,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_VNA = get_raw(fyle)
        time = fyle["raw_data0/A_RX2"].attrs.get('chirp_t')[0]
        amplitude = fyle["raw_data0/A_RX2"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')

    eff_rate = rate/decimation
    if verbose:
        print "\n\nData taken "+str(Dt_tm)
        print "Reported LO is "+str(LO*1e-6)+" MHz"
        print "Reported rate is %f MHz"%(rate/1e6)
        print "Entered decimation is %d"%(decimation)
        print "\tEffective rate is %f kHz"%(eff_rate/1e3)
        print "Reported amplitudes are "+str(amplitude)
        print "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm"

    raw_f = np.arange(f0, f1, (f1-f0)/len(raw_VNA[:,0]))

    array_mean = np.array([])
    for freq in targets:
        print str(raw_f[np.argmin(abs(raw_f-freq))])+' MHz'
        array_mean = np.append(array_mean,raw_VNA[np.argmin(abs(raw_f-freq))])

    return targets, array_mean, Dt_tm, int(time*rate/len(raw_VNA)), time/len(raw_VNA)

def plot_VNA(filename):
    f, z = read_vna(filename)
    crop = 1000
    f = f[crop:-crop]*1e-3
    z = z[crop:-crop]
    f = f[::5]
    z = z[::5]


    resonances, _ = vna_file_fit(filename,[3.468, 3.486, 3.503, 3.505, 3.516, 3.527, 3.539])
    near = .0007
    near_res = []
    for resonance in resonances:
        near_this_res = np.logical_and(f > resonance - near, f < resonance + near)
        if len(near_res) == 0:
            near_res = near_this_res
        else:
            near_res = np.logical_or(near_res,near_this_res)
    plt.figure()
    plt.plot(np.real(z[near_res]),np.imag(z[near_res]),color = 'red',marker ='.',linestyle='',markersize=2)
    plt.plot(np.real(z[np.logical_not(near_res)]),np.imag(z[np.logical_not(near_res)]),marker = '.',linestyle = '',markersize=2)
    plt.title(filename)
    plt.figure()
    plt.plot(f[near_res], 20*np.log10(abs(z[near_res])),color = 'red',marker ='.',linestyle='',markersize=2)
    plt.plot(f[np.logical_not(near_res)],20*np.log10(abs(z[np.logical_not(near_res)])),marker='.',linestyle='',markersize=2)
    plt.title(filename)
    plt.show()

if __name__ == '__main__':
    plot_VNA('USRP_VNA_20200428_123007.h5')
