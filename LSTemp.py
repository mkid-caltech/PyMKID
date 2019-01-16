from __future__ import division
import visa
#import numpy as np
#import time

def setK(nominal_temp):
    # nominal_temp should be in units of K
    rm =  visa.ResourceManager()
    aly2 = rm.open_resource('GPIB0::12::INSTR')
    mode = 1 #PID control
    HRNG = {80.e-3:2, 150.e-3:3, 1.e5:4}
    #nominal_temp =
    if nominal_temp == 0:
        aly2.write("HTRRNG 0")
        aly2.write('SETP {:.3f};'.format(nominal_temp))
    else:
        aly2.write("HTRRNG {}".format(min([v for k,v in HRNG.items() if nominal_temp < k])))
        aly2.write('SETP {:.3f};'.format(nominal_temp))
        aly2.write('CMODE %d' % mode )
    #time.sleep(120)
    #temp = aly2.query('RDGK? 1')
    #print 'Acquiring: {}'.format(temp)
    #aly2.write("HTRRNG 0")
