from __future__ import division
import visa
import numpy as np
import matplotlib.pyplot as plt
import time

def specshot(GPIBnum='13', fcenter=3.5, fspan=1):
    rm =  visa.ResourceManager()
    aly = rm.open_resource('GPIB0::'+GPIBnum+'::INSTR')
    aly.timeout = 25000 #25 seconds
    aly.write('CF {:.9f} GHz;'.format(fcenter))
    aly.write('SP {:.9f} GHz;'.format(fspan))
    time.sleep(1)
    aly.write('RB?;')
    buffr = aly.read()
    RBW = float(buffr)
    print "Bandwidth = "+str(RBW*1E-6)+" MHz"

    aly.write('TA?;')
    buffr = aly.read()
    amplitude = np.loadtxt(buffr.splitlines(), delimiter = ',')
    frequency = np.arange((fcenter-0.5*fspan), (fcenter+0.5*fspan), fspan/len(amplitude))

    plt.figure()
    plt.plot(frequency, amplitude)
    plt.show()

if __name__ == "__main__":
    specshot()
