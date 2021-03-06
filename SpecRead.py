from __future__ import division
import visa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import sys

def specshot(GPIBnum='13', fcenter=3.5, fspan=1, fname=None):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    df = pd.DataFrame()
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
    df["amplitude"] = np.loadtxt(buffr.splitlines(), delimiter = ',')
    df["frequency"] = np.arange((fcenter-0.5*fspan), (fcenter+0.5*fspan), fspan/len(df["amplitude"]))
    df["RBW"] = RBW
    if fname:
        df.to_hdf(fname, key="/{}".format(timestamp))

    plt.figure()
    plt.plot(df["frequency"], df["amplitude"])
    plt.show()

if __name__ == "__main__":
    fcenter_in = float(sys.argv[1])
    fspan_in = float(sys.argv[2])
    fname_in = sys.argv[3]
    specshot(fcenter = fcenter_in, fspan = fspan_in, fname = fname_in)
