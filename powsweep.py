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

def snapshot(aly, fcenter, fspan, averfact=10, points=1601, channel='S21'):
    aly.write('AVERFACT {:d};'.format(averfact))
    aly.write('AVERO ON;')
    aly.write(channel+';')

    aly.write('CENT {:.9f} GHz;'.format(fcenter))
    aly.write('SPAN {:.9f} GHz;'.format(fspan))
    aly.write('OPC?; NUMG {:d};'.format(averfact))

    aly.read()

    aly.write('AUTO;')
    buffr = aly.query('OUTPDATA;')
    zm = np.loadtxt(buffr.splitlines(), delimiter = ',')
    z_data = zm[:,0]+1j*zm[:,1]
    NumPts = points-1
    f_data = np.arange(-(NumPts/2), (NumPts/2)+1, 1)*(fspan/NumPts) + fcenter
    return f_data, z_data

def sweep_pow(fname, pow_list=np.arange(-35, -10, 5), points=1601, chan="S21",  plotit=False):
    pow_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    rm =  visa.ResourceManager()
    aly = rm.open_resource('GPIB0::16::INSTR')
    aly2 = rm.open_resource('GPIB0::12::INSTR')
    #temp = aly2.query('RDGK? 1')
    aly.timeout = 25000 #25 seconds, a 1601 point sweep takes too long for the standard timeout
    aly.write('SOUPON;')
    aly.write('FORM4;')
    aly.write('POIN {:.9f};'.format(points))

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        print fr_list

    fspanray = 1.5e-3*np.ones(len(fr_list)) # GHz, 1.5 MHz

    if plotit == True:
        plt.figure(1)

    for power in pow_list:
        aly.write('POWE {:.2f} DB;'.format(power))
        df = pd.DataFrame()
        for idres, (fcenter, fspan) in enumerate(zip(fr_list, fspanray)):
            temp = aly2.query('RDGK? 1')
            print 'Acquiring: {:.5f} {} {:+.2f}'.format(fcenter,temp,power)
            f_snap, z_snap = snapshot(aly, fcenter, fspan, averfact=10, points=points)
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
    aly = rm.open_resource('GPIB0::16::INSTR')
    aly2 = rm.open_resource('GPIB0::12::INSTR')
    #temp = aly2.query('RDGK? 1')
    aly.timeout = 25000 #25 seconds, a 1601 point sweep takes too long for the standard timeout
    aly.write('SOUPON;')
    aly.write('FORM4;')
    aly.write('POIN {:.9f};'.format(points))

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        print fr_list

    fspanray = 2.0e-3*np.ones(len(fr_list)) # GHz, 2.0 MHz
    fspanray = fspanray/windows # Size of each window in GHz

    if plotit == True:
        plt.figure(1)
    mode = 1 #PID control
    HRNG = {80.e-3:2, 150.e-3:3, 1.e5:4}
    aly.write('POWE {:.2f} DB;'.format(power))
    for nominal_temp in temp_list:
        aly2.write("HTRRNG {}".format(min([v for k,v in HRNG.items() if nominal_temp < k])))
        aly2.write('SETP {:.3f};'.format(nominal_temp))
        aly2.write('CMODE %d' % mode )
        time.sleep(120)
        df = pd.DataFrame()
        for idres, (fcenter, fspan) in enumerate(zip(fr_list, fspanray)):
            temp = aly2.query('RDGK? 1')
            print 'Acquiring: {:.5f} {} {:+.2f}'.format(fcenter,temp,power)
            f_snap = np.array([])
            z_snap = np.array([])
            for window in range(windows):
                f_snap0, z_snap0 = snapshot(aly, fcenter-fspan*((0.5*windows)-window-0.5), fspan, averfact=10, points=points)
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
    aly2.write("HTRRNG 0")
    if plotit == True:
        plt.show()

def plot_pow(fname):
    with h5py.File(fname, "r") as fyle:
        for timestamp in fyle["powsweep"].keys():
            print timestamp
            plt.figure()
            powervals = np.array(fyle["powsweep/"+timestamp].keys())
            powervals = powervals.astype(np.float)
            print sorted(powervals)
            for power in fyle["powsweep/"+timestamp].keys():
                print power
                df = pd.read_hdf(fname, key="powsweep/"+timestamp+"/"+power)
                resID = df['resID']
                f = df['f']
                z = df['z']
                plt.plot(f, 20*np.log10(np.abs(np.array(z))))
                #plt.plot(f[resID==0], 20*np.log10(np.abs(np.array(z[resID==0]))))
            plt.show()

def plot_temp(fname, top = 4):
    with h5py.File(fname, "r") as fyle:
        for timestamp in fyle["tempsweep"].keys():
            print timestamp
            plt.figure()
            for temperature in np.array(fyle["tempsweep/"+timestamp].keys()):
                if temperature.astype(np.float) <= top:
                    print temperature.astype(np.float)
                    df = pd.read_hdf(fname, key="tempsweep/"+timestamp+"/"+temperature)
                    resID = df['resID']
                    f = df['f']
                    z = df['z']
                    plt.plot(f, 20*np.log10(np.abs(np.array(z))))
                    #plt.plot(f[resID==0], 20*np.log10(np.abs(np.array(z[resID==0]))))
            plt.show()
