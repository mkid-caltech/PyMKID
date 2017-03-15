from __future__ import division
import visa
import numpy as np
import math
import cmath
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
    z = zm[:,0]+1j*zm[:,1]
    f = np.arange(-(NumPts/2), (NumPts/2)+1, 1)*(fspan/NumPts) + fcenter
    return f, z

def sweep_pow(fname, chan="S21", rewrite=False):
    rm =  visa.ResourceManager()
    aly = rm.open_resource('GPIB0::16::INSTR')
    aly.write('SOUPON;')
    aly.write('FORM4;')

    with h5py.File(fname, "r+") as fyle:
        f0list = np.array(fyle["{}/f0list".format(chan)])
        if rewrite == True:
            fyle.__delitem__("{}/powsweeps".format(chan))

    fspanray = 1.5e-3*np.ones(len(f0list))
    pows = np.arange(-30, -12, 2)

    for idpow in range(len(pows)):
        power = pows[idpow]
        for idres in range(len(f0list)):
            fcenter = f0list[idres]
            fspan = fspanray[idres]
            f, z = snapshot(aly, fcenter, fspan, power);
            with h5py.File(fname, "r+") as fyle:
                fyle["{}/powsweeps/".format(chan)+"{:.5f}/".format(f0list[idres])+"_{}_f".format(abs(pows[idpow]))] = f
                fyle["{}/powsweeps/".format(chan)+"{:.5f}/".format(f0list[idres])+"_{}_z".format(abs(pows[idpow]))] = z
