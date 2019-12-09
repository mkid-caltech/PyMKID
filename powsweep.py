from __future__ import division
import visa
import numpy as np
#import math
#import cmath
import matplotlib.pyplot as plt
import h5py
import re
import pandas as pd
import time
import datetime
#import time

def snapshot(VNA, fcenter, fspan, averfact=10, points=1601, channel='S21'):
    VNA.write('AVERFACT {:d};'.format(averfact))
    VNA.write('AVERO ON;')
    VNA.write(channel+';')

    VNA.write('CENT {:.9f} GHz;'.format(fcenter))
    VNA.write('SPAN {:.9f} GHz;'.format(fspan))
    VNA.write('OPC?; NUMG {:d};'.format(averfact))

    VNA.read()

    VNA.write('AUTO;')
    buffr = VNA.query('OUTPDATA;')
    zm = np.loadtxt(buffr.splitlines(), delimiter = ',')
    z_data = zm[:,0]+1j*zm[:,1]
    NumPts = points-1
    f_data = np.arange(-(NumPts/2), (NumPts/2)+1, 1)*(fspan/NumPts) + fcenter
    return f_data, z_data

def sweep_pow(fname, pow_list=np.arange(-35, -10, 5), points=1601, chan="S21",  plotit=False):
    pow_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    rm =  visa.ResourceManager()
    VNA = rm.open_resource('GPIB0::16::INSTR')
    LakeShore = rm.open_resource('GPIB0::12::INSTR')
    #temp = LakeShore.query('RDGK? 1')
    VNA.timeout = 25000 #25 seconds, a 1601 point sweep takes too long for the standard timeout
    VNA.write('SOUPON;')
    VNA.write('FORM4;')
    VNA.write('POIN {:.9f};'.format(points))

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        print fr_list

    fspanray = 1.5e-3*np.ones(len(fr_list)) # GHz, 1.5 MHz

    if plotit == True:
        plt.figure(1)

    for power in pow_list:
        VNA.write('POWE {:.2f} DB;'.format(power))
        df = pd.DataFrame()
        for idres, (fcenter, fspan) in enumerate(zip(fr_list, fspanray)):
            temp = LakeShore.query('RDGK? 1')
            print 'Acquiring: {:.5f} {} {:+.2f}'.format(fcenter,temp,power)
            f_snap, z_snap = snapshot(VNA, fcenter, fspan, averfact=10, points=points)
            if plotit == True:
                plt.plot(f_snap, 20*np.log10(np.abs(np.array(z_snap))))
            df_temp = pd.DataFrame()
            df_temp["f"] = f_snap
            df_temp["z"] = z_snap
            df_temp["T"] = temp
            df_temp["f0"] = fcenter
            df_temp["resID"] = idres
            df_temp["power"] = power
            df = df.append(df_temp)
            # if rewrite == True:
            #     with h5py.File(fname, "r+") as fyle:
            #         fyle["{:.5f}/{:.5f}/{:+.2f}/f".format(fcenter,temp,power)] = f_snap
            #         fyle["{:.5f}/".format(fr_list[idres])+"{:.5f}/".format(temp)+"{:+.2f}/".format(power)+"z"] = z_snap
        df.to_hdf(fname, key="/powsweep/{}/{:+.2f}".format(pow_timestamp,power))
    if plotit == True:
        plt.show()

def sweep_temp(fname, power, temp_list=1E-3*np.arange(70, 150, 5), points=1601, chan="S21",  plotit=False, windows=1):
    if max(temp_list) >= 2:
        print "Max temperature in temp_list is greater than 2K. Cancelling."
        raise SystemExit
    temp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    rm =  visa.ResourceManager()
    VNA = rm.open_resource('GPIB0::16::INSTR')
    LakeShore = rm.open_resource('GPIB0::12::INSTR')
    #temp = LakeShore.query('RDGK? 1')
    VNA.timeout = 25000 #25 seconds, a 1601 point sweep takes too long for the standard timeout
    VNA.write('SOUPON;')
    VNA.write('FORM4;')
    VNA.write('POIN {:.9f};'.format(points))

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        print fr_list

    fspanray = 2.0e-3*np.ones(len(fr_list)) # GHz, 2.0 MHz
    fspanray = fspanray/windows # Size of each window in GHz

    if plotit == True:
        plt.figure(1)
    mode = 1 #PID control
    HRNG = {80.e-3:2, 150.e-3:3, 1.e5:4}
    VNA.write('POWE {:.2f} DB;'.format(power))
    for nominal_temp in temp_list:
        LakeShore.write("HTRRNG {}".format(min([v for k,v in HRNG.items() if nominal_temp < k])))
        LakeShore.write('SETP {:.3f};'.format(nominal_temp))
        LakeShore.write('CMODE %d' % mode )
        time.sleep(120)
        df = pd.DataFrame()
        for idres, (fcenter, fspan) in enumerate(zip(fr_list, fspanray)):
            temp = LakeShore.query('RDGK? 1')
            print 'Acquiring: {:.5f} {} {:+.2f}'.format(fcenter,temp,power)
            f_snap = np.array([])
            z_snap = np.array([])
            for window in range(windows):
                f_snap0, z_snap0 = snapshot(VNA, fcenter-fspan*((0.5*windows)-window-0.5), fspan, averfact=10, points=points)
                f_snap = np.append(f_snap,f_snap0)
                z_snap = np.append(z_snap,z_snap0)
            if plotit == True:
                plt.plot(f_snap, 20*np.log10(np.abs(np.array(z_snap))))
            df_temp = pd.DataFrame()
            df_temp["f"] = f_snap
            df_temp["z"] = z_snap
            df_temp["T"] = temp
            df_temp["f0"] = fcenter
            df_temp["resID"] = idres
            df_temp["power"] = power
            df = df.append(df_temp)
            # if rewrite == True:
            #     with h5py.File(fname, "r+") as fyle:
            #         fyle["{:.5f}/{:.5f}/{:+.2f}/f".format(fcenter,temp,power)] = f_snap
            #         fyle["{:.5f}/".format(fr_list[idres])+"{:.5f}/".format(temp)+"{:+.2f}/".format(power)+"z"] = z_snap
        df.to_hdf(fname, key="/tempsweep/{}/{:+.3f}".format(temp_timestamp,nominal_temp)) # Causes complaints about invalid Python identifiers ("2018-10-29-09-10" and "+0.07")
    LakeShore.write("HTRRNG 0")
    if plotit == True:
        plt.show()

def sweep_temp_pow(fname, temp_list=1E-3*np.arange(70, 150, 5), pow_list=np.arange(-35, -10, 5), points=1601, chan="S21",  plotit=False, windows=1):
    if max(temp_list) >= 2:
        print "Max temperature in temp_list is greater than 2K. Cancelling."
        raise SystemExit
    temp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    rm =  visa.ResourceManager()
    VNA = rm.open_resource('GPIB0::16::INSTR')
    LakeShore = rm.open_resource('GPIB0::12::INSTR')
    #temp = LakeShore.query('RDGK? 1')
    VNA.timeout = 25000 #25 seconds, a 1601 point sweep takes too long for the standard timeout
    VNA.write('SOUPON;')
    VNA.write('FORM4;')
    VNA.write('POIN {:.9f};'.format(points))

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        print fr_list

    fspanray = 2.0e-3*np.ones(len(fr_list)) # GHz, 2.0 MHz
    fspanray = fspanray/windows # Size of each window in GHz

    if plotit == True:
        plt.figure(1)
    mode = 1 #PID control
    HRNG = {80.e-3:2, 150.e-3:3, 1.e5:4}
    for nominal_temp in temp_list:
        LakeShore.write("HTRRNG {}".format(min([v for k,v in HRNG.items() if nominal_temp < k])))
        LakeShore.write('SETP {:.3f};'.format(nominal_temp))
        LakeShore.write('CMODE %d' % mode )
        time.sleep(120)
        for power in pow_list:
            VNA.write('POWE {:.2f} DB;'.format(power))
            df = pd.DataFrame()
            for idres, (fcenter, fspan) in enumerate(zip(fr_list, fspanray)):
                temp = LakeShore.query('RDGK? 1')
                print 'Acquiring: {:.5f} {} {:+.2f}'.format(fcenter,temp,power)
                f_snap = np.array([])
                z_snap = np.array([])
                for window in range(windows):
                    f_snap0, z_snap0 = snapshot(VNA, fcenter-fspan*((0.5*windows)-window-0.5), fspan, averfact=10, points=points)
                    f_snap = np.append(f_snap,f_snap0)
                    z_snap = np.append(z_snap,z_snap0)
                if plotit == True:
                    plt.plot(f_snap, 20*np.log10(np.abs(np.array(z_snap))))
                df_temp = pd.DataFrame()
                df_temp["f"] = f_snap
                df_temp["z"] = z_snap
                df_temp["T"] = temp
                df_temp["f0"] = fcenter
                df_temp["resID"] = idres
                df_temp["power"] = power
                df = df.append(df_temp)
                # if rewrite == True:
                #     with h5py.File(fname, "r+") as fyle:
                #         fyle["{:.5f}/{:.5f}/{:+.2f}/f".format(fcenter,temp,power)] = f_snap
                #         fyle["{:.5f}/".format(fr_list[idres])+"{:.5f}/".format(temp)+"{:+.2f}/".format(power)+"z"] = z_snap
            df.to_hdf(fname, key="/tempsweep/{}/{:+.3f}/{:+.2f}".format(temp_timestamp,nominal_temp,power)) # Causes complaints about invalid Python identifiers ("2018-10-29-09-10" and "+0.07")
    LakeShore.write("HTRRNG 0")
    if plotit == True:
        plt.show()

def plot_pow(fname):
    with h5py.File(fname, "r") as fyle:
        for timestamp in fyle["powsweep"].keys():
            print timestamp
            plt.figure()
            powernams = np.array(fyle["powsweep/"+timestamp].keys())
            powervals = powernams.astype(np.float)
            #print sorted(powervals)
            #print powervals.argsort()
            powernams = powernams[powervals.argsort()]
            for power in powernams:
                print power
                df = pd.read_hdf(fname, key="powsweep/"+timestamp+"/"+power)
                resID = df['resID']
                f = df['f']
                z = df['z']
                plt.plot(f, 20*np.log10(np.abs(np.array(z))), label=power)
                #plt.plot(f[resID==0], 20*np.log10(np.abs(np.array(z[resID==0]))))
            plt.legend()
            plt.show()

def plot_temp(fname, top=4, together=False):
    with h5py.File(fname, "r") as fyle:
        for timestamp in fyle["tempsweep"].keys():
            print timestamp
            plt.figure(1)
            temperatures = np.array(fyle["tempsweep/"+timestamp].keys())
            temperatures = temperatures[temperatures!='MB']
            temperatures = temperatures[temperatures!='RES']
            for temperature in temperatures:
                if temperature.astype(np.float) <= top:
                    print temperature.astype(np.float)
                    df = pd.read_hdf(fname, key="tempsweep/"+timestamp+"/"+temperature)
                    resID = df['resID']
                    f = df['f']
                    z = df['z']
                    plt.plot(f, 20*np.log10(np.abs(np.array(z))), label=temperature)
                    #plt.plot(f[resID==0], 20*np.log10(np.abs(np.array(z[resID==0]))))
            if not together:
                plt.legend()
                plt.show()
        if together:
            plt.legend()
            plt.show()
