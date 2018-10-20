from __future__ import division
import visa
import numpy as np
#import math
#import cmath
import matplotlib.pyplot as plt
import h5py
from powsweep import snapshot
#import datetime

def freqsweep(aly, fwinstart, fwinend, fwinsize, power, averfact, channel="S21", plotting=False):
    """
    freqsweep preforms a windowed frequency sweep using the HP VNA
    Input parameters:
        aly: an instence of a PyVISA instrument object that represents the VNA
        fwinstart: center of the first window
        fwinend: center of the last window
        fwinsize: window step
        power: input power
        averfact: number of averages per window
        channel: scattering parameter to measure (S21, S11 ect...)
        plotting: whether or not to display plots while data is taken

    Returns:
        tuple (f, z) where
        f: frequencys
        z: complex impedence
    """
    # choose number of points per window
    NumPts = 200 # actual number of points is 201
    fwinres = fwinsize/NumPts

    # set power, averaging number, channel, and turn averaging on,
    aly.write('POWE {:f} DB;'.format(power))
    aly.write('AVERFACT {:d};'.format(averfact))
    aly.write('AVERO ON;')
    aly.write(channel+';')

    # initialize f and z
    f = []
    z = []

    # open plots if plotting
    if plotting:
        plt.figure(channel)
        plt.ion()
        plt.title(channel)

    # loop over each window
    for i in range(int((fwinend-fwinstart)/fwinsize)+1):
        # define the window center and range of points to be taken
        fwincenter = fwinstart+(i*fwinsize)
        fwin = np.arange(-(NumPts/2),(NumPts/2)+1,1) * fwinres + fwincenter

        aly.write('{:.3f} '.format(fwincenter)) # don't know what this does (if anything?)

        # write center, window span, and NUMG? to VNA
        aly.write('CENT {:.9f} GHz;'.format(fwincenter))
        aly.write('SPAN {:.9f} GHz;'.format(fwinsize))
        aly.write('OPC?; NUMG {:d};'.format(averfact))
        aly.read()
        aly.write('AUTO;')
        # ask for the output
        aly.write('OUTPDATA;')
        # record the data
        buffr = aly.read()
        # interpret the data
        zm = np.loadtxt(buffr.splitlines(), delimiter = ',')
        zwin = zm[:,0]+1j*zm[:,1]
        # add data to f and z arrays
        f.extend(fwin)
        z.extend(zwin)
        # add data to plots if plotting
        if plotting:
            plt.plot(np.array(f), 20*np.log10(np.abs(np.array(z))), 'c')
            plt.pause(0.02)

    return np.array(f), np.array(z)

def freqsweep1(aly, fwinstart, fwinend, fwinsize, power, averfact, channel="S21", plotting=False):
    """
    freqsweep preforms a windowed frequency sweep using the HP VNA
    Input parameters:
        aly: an instence of a PyVISA instrument object that represents the VNA
        fwinstart: center of the first window
        fwinend: center of the last window
        fwinsize: window step
        power: input power
        averfact: number of averages per window
        channel: scattering parameter to measure (S21, S11 ect...)
        plotting: whether or not to display plots while data is taken

    Returns:
        tuple (f, z) where
        f: frequencys
        z: complex impedence
    """
    # initialize f and z
    f = []
    z = []

    # open plots if plotting
    if plotting:
        plt.figure(channel)
        plt.ion()
        plt.title(channel)

    # loop over each window
    for i in range(int((fwinend-fwinstart)/fwinsize)+1):
        # define the window center
        fwincenter = fwinstart+(i*fwinsize)

        # use powsweep.snapshot to take a sweep over the current window
        fwin, zwin = snapshot(aly, fwincenter, fwinsize, power, averfact=averfact, points=201, channel=channel)

        # add data to f and z arrays
        f.extend(fwin)
        z.extend(zwin)
        # add data to plots if plotting
        if plotting:
            plt.plot(np.array(f), 20*np.log10(np.abs(np.array(z))), 'c')
            plt.pause(0.02)

    return np.array(f), np.array(z)

def save_scatter(fname, fwinstart=1, fwinend=5, fwinsize=0.01, power=-40, averfact=10, channels=["S21", "S22", "S12", "S11"], GPIBnum=16, plotting=False):
    """
    uses freqsweep to preform a windowed frequency sweep on HP VNA channels
    Input parameters:
        fname: desired name for the output file. eg "YY160622.h5"
        fwinstart: center of the first window
        fwinend: center of the last window
        fwinsize: window step
        power: input power
        averfact: number of averages per window
        channels: list of channels to probe
        GPIBnum: number corresponding to the VNA GPIB
        plotting: whether or not to display plots while data is taken

    Returns:
        file fname with f and z for each channel probed
    """
    # open GPIB connection to the VNA
    rm =  visa.ResourceManager()
    #aly = rm.open_resource('GPIB0::16::INSTR')
    aly = rm.open_resource("GPIB0::{}::INSTR".format(GPIBnum))

    # make sure VNA is on and communicating in the correct ascii form
    aly.write('SOUPON;')
    aly.write('FORM4;')

    # create and open file
    with h5py.File(fname, "w") as fyle:
        for chan in channels:
            # use freqsweep to take a sweep for each channel
            f, z = freqsweep(aly, fwinstart, fwinend, fwinsize, power, averfact, chan, plotting=plotting)
            # save f and z to file fname
            fyle["{}/f".format(chan)] = f
            fyle["{}/z".format(chan)] = z

def replot(fname, channels=["S21", "S22", "S12", "S11"], color='b', legend=False, style="-", linewidth=1):
    """
    170323 Yen-Yung Chang
    add style and linewidth arguments
    """
    """
    16xxxx Taylor
    replots the graphs shown during save_scatter(plotting=True) for file fname
    Input parameters:
        fname: desired name for the input file. eg "YY160622.h5". Should be made using save_scatter
        channels: list of channels to plot. Each must have been probed during save_scatter

    Returns:
        transmission vs frequency plot for each channel
    """
    # interactive plot
    #plt.ion()

    # open and read file
    with h5py.File(fname, "r") as fyle:
        for chan in channels:
            f = fyle["{}/f".format(chan)]
            z = fyle["{}/z".format(chan)]
            zt = 20*np.log10(np.abs(np.array(z)))
            # make plots
            plt.figure(chan)
            plt.title(chan)
            plt.xlabel('GHz')
            plt.ylabel('dB')
            line_style = color+style
            plt.plot(np.array(f), zt, line_style, label=fname, linewidth=linewidth)
            if legend:
                plt.legend(loc="best")
            #plt.pause(0.02)
    plt.show()
