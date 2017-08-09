from __future__ import division
import visa
import numpy as np
#import math
#import cmath
import matplotlib.pyplot as plt
import h5py

def snapshot(aly, fcenter, fspan, power, averfact=1, points=1601, channel='S21'):
    NumPts = points-1
    aly.write(channel+';')
    aly.write('POIN {:d};'.format(points))

    aly.write('POWE {:.2f} DB;'.format(power))
    aly.write('CENT {:.9f} GHz;'.format(fcenter))
    aly.write('SPAN {:.9f} GHz;'.format(fspan))

    aly.write('AVERFACT {:d};'.format(averfact))
    aly.write('AVERO ON;')
    aly.write('OPC?; NUMG {:d};'.format(averfact))
    aly.read()

    aly.write('AUTO;')
    buffr = aly.query('OUTPDATA;')
    zm = np.loadtxt(buffr.splitlines(), delimiter = ',')
    z_data = zm[:,0]+1j*zm[:,1]
    f_data = np.arange(-(NumPts/2), (NumPts/2)+1, 1)*(fspan/NumPts) + fcenter
    return f_data, z_data

def sweep_pow(fname, chan="S21", rewrite=False, plotit=False):
    rm =  visa.ResourceManager()
    aly = rm.open_resource('GPIB0::16::INSTR')
    aly.write('SOUPON;')
    aly.write('FORM4;')

    with h5py.File(fname, "r+") as fyle:
        fr_list = np.array(fyle["{}/fr_list".format(chan)])
        if rewrite == True:
            if "{}/powsweeps".format(chan) in fyle:
                fyle.__delitem__("{}/powsweeps".format(chan))

    fspanray = 1.5e-3*np.ones(len(fr_list))
    pows = np.arange(5, 6, 15)

    if plotit == True:
        plt.figure()

    for idpow in range(len(pows)):
        power = pows[idpow]
        for idres in range(len(fr_list)):
            fcenter = fr_list[idres]
            fspan = fspanray[idres]
            f_snap, z_snap = snapshot(aly, fcenter, fspan, power, averfact=10)
            if plotit == True:
                plt.plot(f_snap, 20*np.log10(np.abs(np.array(z_snap))))
            if rewrite == True:
                with h5py.File(fname, "r+") as fyle:
                    fyle["{}/powsweeps/".format(chan)+"{:.5f}/".format(fr_list[idres])+"_{}_f".format(abs(pows[idpow]))] = f_snap
                    fyle["{}/powsweeps/".format(chan)+"{:.5f}/".format(fr_list[idres])+"_{}_z".format(abs(pows[idpow]))] = z_snap

    if plotit == True:
        plt.show()
