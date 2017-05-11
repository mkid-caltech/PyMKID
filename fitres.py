from __future__ import division
#import visa
import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import math
import cmath
import matplotlib.pyplot as plt
import h5py
from functools import partial

def removecable(f, z, tau, f1):
    z1 = np.array(z)*np.exp(-2j*np.pi*(np.array(f)-f1)*tau)
    return z1

def estpara(f, z):
    """
    returns:
        f0g:  The estimated center frequency for this resonance
        Qg:   The estimated total quality factor
        idf0: The estimated center frequency in index number space
        iddf: Half the 3dB bandwidth in index number space
    """

    z1 = np.mean(z[:10])
    z2 = np.mean(z[-10:])
    zr = np.mean([z1, z2])

    idf0 = max(np.argmax(abs(np.array(z)-zr)),1)

    z0 = z[idf0]
    f0g = f[idf0]
    rg = abs(zr - z0)/2
    zleft = z[:idf0]
    zright = z[idf0:]

    id3db1 = np.argmin(abs(abs(abs(zleft-zr)) - np.sqrt(2)*rg))
    id3db2 = np.argmin(abs(abs(abs(zright-zr)) - np.sqrt(2)*rg))
    iddf = max(abs(idf0 - id3db1), abs(id3db2))
    Qg = f0g/(2*iddf*(f[1]-f[0]))

    return f0g, Qg, idf0, iddf

def circle2(z):
    # == METHOD 2b ==
    # "leastsq with jacobian"
    x = z.real
    y = z.imag
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_estimate = x_m, y_m
    center_2b, ier = opt.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b              # circle center
    Ri_2b = calc_R(*center_2b)            # distance of each data point from center_2b
    R_2b = Ri_2b.mean()                   # average Ri_2b, used as predicted radius
    residu_2b = sum((Ri_2b - R_2b)**2)    # residual?

    zc =  center_2b[0]+center_2b[1]*1j
    r = R_2b
    residue = residu_2b

    plt.figure(2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(x, y, '.')
    t = np.arange(0,2*np.pi,0.002)
    xcirc = center_2b[0]+r*np.cos(t)
    ycirc = center_2b[1]+r*np.sin(t)
    plt.plot(xcirc,ycirc)

    return residue, zc, r

def unfoldphasenoisy(phain, pct=0):
    # pct is the % of point to be averaged, set to negative to go through every point (no average)
    numpts = len(phain)
    numsmopts = int(math.floor(pct*numpts))
    numsmopts = max(1, numsmopts)

    phaout = np.array(phain)

    for j in range(len(phaout)-1):
        oldave = 0
        if j < numsmopts:
            oldave = np.mean(phaout[:j+1])
        else:
            oldave = np.mean(phaout[j-numsmopts:j+1])
        valsel = [phaout[j+1]-2*np.pi, phaout[j+1], phaout[j+1]+2*np.pi]
        idd = np.argmin(abs(valsel-oldave))
        phaout[j+1:] = (phaout[j+1:] + (idd-1)*2*np.pi)

    if phaout[0] < -np.pi:
        phaout = phaout+2*np.pi

    return phaout

def smooth(x, window_len=7):
    # smooths an input array using a moving average of size window_len
    y=[]
    winside = int((window_len-1)/2)
    for i in range(len(x)):
        if (i-winside)<0 and (i+winside)<len(x):
            meand = x[:(2*i)+1]
        if (i-winside)>=0 and (i+winside)>=len(x):
            meand = x[len(x)-(2*(len(x)-i))+1:]
        if (i-winside)>=0 and (i+winside)<len(x):
            meand = x[i-winside:i+winside+1]
        if (i-winside)<0 and (i+winside)>=len(x):
            if (winside-i)>(i+winside-len(x)):
                meand = x[:(2*i)+1]
            if (winside-i)<(i+winside-len(x)):
                meand = x[len(x)-(2*(len(x)-i))+1:]
            if (winside-i)==(i+winside-len(x)):
                meand = x
        y.append([np.mean(meand)])
    return np.array(y)

def fid(f, f0):
    loci = np.argmin(abs(f-f0))
    return loci

def trimdata(f, z, f0, Q, numspan=1):
    f0id = fid(f, f0)
    idspan =  (f0/Q)/(f[1]-f[0])
    idstart = int(math.floor(max(f0id-(idspan*numspan), 0)))
    idstop  = int(math.floor(min(f0id+(idspan*numspan), len(f))))
    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return fb, zb

def phasefunc(fff, Qff, f0ff, phiff):
    return -phiff+2*np.arctan(2*Qff*(1-fff/f0ff))

def fitphase(f, z, f0g, Qg, numspan=1):
    ang = unfoldphasenoisy(np.angle(z, deg=False))
    fstep = f[1]-f[0]
    #while np.mean(ang)>np.pi:          seem unnecessary
    #    ang = ang -2*np.pi
    #while np.mean(ang)<-np.pi:
    #    ang = ang +2*np.pi

    # estimate f0, Q and phi
    hnumsmopts = int(math.floor(((f0g/Qg)/3)/fstep))
    hnumsmopts = int(max(min(hnumsmopts, 20), 1))
    zm = smooth(z, window_len=hnumsmopts*2)
    angm = unfoldphasenoisy(np.angle(zm, deg=False))      # this filter also does nothing to the current data set
    dangm = angm[hnumsmopts:] - angm[:-hnumsmopts]
    #fd    =    f[hnumsmopts:] -    f[:-hnumsmopts]
    idd = np.argmin(dangm)
    f0 = f[idd + hnumsmopts]

    # linear fit around f0 to find Q and phi
    idh = idd + hnumsmopts
    wid = int(math.floor(((f0/Qg)/3)/fstep))
    fmid = f[max(idh-wid,0):min(idh+wid,len(f))]
    xmid = (f0-fmid)/f0
    angmid = ang[max(idh-wid,0):min(idh+wid,len(f))]

    # using polyfit instead of mregress.m
    cof = np.polyfit(xmid, angmid,1)  # angmid=(cof[0]*xmid)+cof[1]
    phi = -cof[1];
    Q = abs(cof[0]/4)

    # set the width of the weighing function to 2df
    ft, angt = trimdata(f, ang, f0, Q, numspan=numspan)
    #ft, zt   = trimdata(f, z, f0, Q, numspan=numspan)

    # using robust(?) fit from curve fit
    fresult = opt.curve_fit(phasefunc, ft, angt, p0=[Q, f0, phi])

    return fresult[0], ft

def roughfit(f, z, tau0, numspan=1):
    f1 = f[int(math.floor(len(f)/2))]

    # remove cable term
    z1 = removecable(f, z, tau0, f1)

    # estimate f0, Q (very rough)
    f0g, Qg, idf0, iddf = estpara(f,z1)

    # fit circle using data points f0g +- 2*f0g/Qg
    id1 = max(idf0-(2*iddf), 0)
    id2 = min(idf0+(2*iddf), len(f))

    if len(range(id1, id2)) < 10:
        id1 = 0
        id2 = len(f)

    residue, zc, r = circle2(z1[id1:id2])

    # rotation and traslation to center
    z2 = (zc - z1)*np.exp(-1j*np.angle(zc, deg=False))
    plt.figure(3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(z2.real, z2.imag, '.')

    # trim data and fit phase
    fresult, ft = fitphase(f, z2, f0g, Qg, numspan=numspan)
    idft1 = fid(f, ft[0])
    widft = len(ft)
    zt = z[idft1:idft1+widft] #-1
    ft = f[idft1:idft1+widft] #-1

    # result
    Q = fresult[0]
    f0 = fresult[1]
    phi = fresult[2]

    zd = -zc/abs(zc)*np.exp(-1j*phi)*r*2
    zinf = zc - zd/2

    Qc = (abs(zinf)/abs(zd))*Q
    #Qc = Q*(abs(zc)+r)/(2*r)
    Qi = 1/((1/Q)-(1/Qc))

    zf0 = zinf + zd

    # projection of zf0 into zinf
    l = np.array(zf0*np.conj(zinf)).real/abs(zinf)**2

    # Q/Qi0 = l
    Qi0 = Q/l

    return f1, zc, r, ft, zt, f0, Q, phi, zd, zinf, Qc, Qi, Qi0

def estsig(f, z):
    # seems to be for estimating variance (sigma squared)
    sig2 = np.mean(abs(np.diff(z[:int(len(z)/5)]))**2)/2
    sig2x = sig2/2
    return sig2x

def resfunc(f, f0, Q, zdx, zdy, zinfx, zinfy, tau, f1=None):
    zd = zdx+1j*zdy
    zinf = zinfx+1j*zinfy
    #Q=Q*10000

    ff = abs(f)
    yy = np.exp(2j*np.pi*(ff-f1)*tau)*(zinf+(zd/(1+2j*Q*(ff-f0)/f0)))

    yy = np.array(yy)
    ayy = yy.real
    ayy[f>0] = 0
    byy = yy.imag
    byy[f<0] = 0
    y = ayy+byy
    return y

def resfunc2(f, fr, Qr, Qc, areal, aimag, phi0, tau, f1=None):
    x = abs(f)
    a = areal + 1j*aimag
    complexy = a*np.exp(2j*np.pi*(x-f1)*tau)*(1-(((Qr/Qc)*np.exp(1j*phi0))/(1+(2j*Qr*(x-fr)/fr))))

    complexy = np.array(complexy)
    realy = complexy.real
    realy[f>0] = 0
    imagy = complexy.imag
    imagy[f<0] = 0
    y = realy+imagy
    return y

def resfunc3(f, fr, Qr, Qc, a, phi, tau):
    y = a*np.exp(2j*np.pi*f*tau)*(1-(((Qr/Qc)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return y

def finefit(f, z, tau0, numspan=1):
    """
    finefit fits f and z to the resonator model described in Jiansong's thesis
    Input parameters:
        f: frequencys
        z: complex impedence
        tau0: cable delay
        numspan: how many bandwidths (f0/Q) to cut around the data during fit (numspan=1 gives f0/Q on each side for a total width of 2 bandwidth)

    Returns:
        f0: frequency center of the resonator
        Q: total Q factor of the resonator
        Qi0: unsure?
            Qi is the internal Q factor of the resonator. Accounts for all the other loss channels
            (Ql, QTL) than through coupling to the feedline (Qc)[paraphrased from Jiansong thesis]
        Qc: coupling Q factor of the resonator. Caused be coupling to the feedline
        zc:
    """
    # find starting parameters using a rough fit
    f1, zc, r, ft, zt, f0, Q, phi, zd, zinf, Qc, Qi, Qi0 = roughfit(f, z, tau0, numspan=numspan)

    # estimate variance
    sig2x = estsig(f,z)

    # trim data
    fnew, znew = trimdata(f, z, f0, Q, numspan=numspan)
    if len(fnew)>10:
        f = fnew
        z = znew
        f1 = f[int(math.floor(len(f)/2))]

    # the fit produced using the starting parameters
    resctest = 1j*resfunc(f, f0, Q, zd.real, zd.imag, zinf.real, zinf.imag, tau0, f1=f1) + resfunc(-f, f0, Q, zd.real, zd.imag, zinf.real, zinf.imag, tau0, f1=f1)

    # combine x and y data so the fit can go over both simultaneously
    xdata = np.concatenate([-f, f])
    ydata = np.concatenate([z.real, z.imag])

    # separate the real and imaginary parts of zd and zinf
    zdx = zd.real
    zdy = zd.imag
    zinfx = zinf.real
    zinfy = zinf.imag

    # perform the fine fit
    #fparams, fcov = opt.curve_fit(partial(resfunc, f1=f1), xdata, ydata, p0=[f0, Q, zdx, zdy, zinfx, zinfy, tau0])

    a1 = zinf # a1=a*np.exp(+2j*np.pi*f1*tau)
    phi0 = np.angle(-zd/zinf, deg=False)
    fparams, fcov = opt.curve_fit(partial(resfunc2, f1=f1), xdata, ydata, p0=[f0, Q, Qc, a1.real, a1.imag, phi0, tau0])

    # untangle best fit parameters
    #f0 = fparams[0]
    #Q = fparams[1]
    #zd = fparams[2]+1j*fparams[3]
    #zinf = fparams[4]+1j*fparams[5]
    #tau = fparams[6]

    fr_fine = fparams[0]
    Qr_fine = fparams[1]
    Qc_fine = fparams[2]
    a_fine = (fparams[3] + 1j*fparams[4])*np.exp(-2j*np.pi*f1*fparams[6])
    phi_fine = fparams[5]
    tau_fine = fparams[6]

    # find Qc, unsure where this equation originates
    #Qc = (abs(zinf)/abs(zd))*Q
    # find Qi (all Q that isn't Qc)
    Qi_fine = 1/((1/Qr_fine)-(1/Qc_fine))


    # the next section finds Qi0 and other parameters. I'm not sure where any of these equations come from or why we want these parameters
    zf0 = zinf + zd
    # projection of zf0 into zinf
    l = (zf0*np.conj(zinf)).real/abs(zinf)**2
    # Q/Qi0 = l
    Qi0 = Q/l
    zc = zinf + zd/2
    r = abs(zd/2)
    phi = np.angle(-zd/zc, deg=False)

    # plot the rough and fine fits to compare
    plt.figure(4)

    # the fit produced using the best parameters
    #resftest = 1j*resfunc2(f, fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5], fparams[6], f1=f1) + resfunc2(-f, fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5], fparams[6], f1=f1)
    resftest = resfunc3(f, fr_fine, Qr_fine, Qc_fine, a_fine, phi_fine, tau_fine)

    # plot rough fit in red
    plt.plot(f, 20*np.log10(np.abs(resctest)), 'r')

    # plot fine fit in blue
    plt.plot(f, 20*np.log10(np.abs(resftest)), 'b')

    # plot real data
    plt.plot(f, 20*np.log10(np.abs(z)), '.')

    #return f0, Q, Qi0, Qc, zc
    return fr_fine, Qr_fine, Qc_fine, a_fine, phi_fine, tau_fine

def sweep_fit(fname, nsig=3, fwindow=5e-4, chan="S21", write=False):
    """
    sweep_fit fits data taken using save_scatter to the resonator model described in Jiansong's thesis
    Input parameters:
        fname: name of the file containing save_scatter data. eg "YY160622.h5"
        nsig: number of sigma above which a peak is considered a resonator peak
        fwindow: half width of the window cut around each resonator before the resonator is fit
        chan: channel from save_scatter being analyzed
        write: whether or not sweep_fit will save its fit data to the fname file

    Returns:
        f0list, Qlist, Qilist, Qclist (lists of values for each resonator) save to the fname file
    """
    # open file and read data
    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])

    # Butterworth filter to remove features larger than n*nfreq
    nfreq = fwindow/(f[-1]-f[0])    # The 
    b, a = sig.butter(2, 3*nfreq, btype='highpass')

    # Zero phase digital filter
    az = abs(z)
    azn = -sig.filtfilt(b, a, az)

    # set all negative azn values to zero
    low_values_indices = azn < 0
    azn[low_values_indices] = 0

    # plot azn with a horizontal line to show which peaks are above nsig sigma
    plt.figure(1)
    fig, axarr = plt.subplots(nrows=2, sharex=True, num=1)
    axarr[0].plot(f, 20*np.log10(np.abs(np.array(z)))) # plot the unaltered transmission
    axarr[0].set_title('Sharing X axis')
    axarr[1].plot(f, azn)
    bstd = np.std(azn)
    curr_xlims = [min(f), max(f)]
    nsigline =  bstd*nsig*np.array([1, 1])
    axarr[1].plot(curr_xlims, nsigline, 'r-')

    # initialize peaklist
    peaklist = np.array([], dtype = int)

    # initialize mn above max and mx below min
    mn = max(azn)+np.inf
    mx = -np.inf
    mnpos = np.nan
    mxpos = np.nan

    lookformax = True
    delta = nsig*bstd

    # find peaks and add them to peaklist
    for i in range(len(azn)):
        cp = azn[i]
        if cp >= mx:
            mx = cp
            mxpos = f[i]
        if cp <= mn:
            mn = cp
            mnpos = f[i]
        if lookformax == True:
            if cp < (mx-delta):
                fpos = np.argmin(abs(f-mxpos))
                peaklist = np.append(peaklist, fpos)
                mn = cp
                mnpos = f[i]
                lookformax = False
        else:
            if cp > (mn+delta):
                mx = cp
                mxpos = f[i]
                lookformax = True

    # plot the peak points from peaklist
    axarr[1].plot(f[peaklist], azn[peaklist], 'gx', label=str(len(peaklist))+" resonances identified")
    plt.legend()
    plt.show()

    # initialize the f0, Q, Qi, and Qc parameter lists
    f0list = np.zeros(len(peaklist))
    Qlist = np.zeros(len(peaklist))
    Qilist = np.zeros(len(peaklist))
    Qclist = np.zeros(len(peaklist))
    zclist = np.array([0.+0j]*len(peaklist))

    fr_list = np.zeros(len(peaklist))
    Qr_list = np.zeros(len(peaklist))
    Qc_list = np.zeros(len(peaklist))
    a_list = np.array([0.+0j]*len(peaklist))
    phi_list = np.zeros(len(peaklist))
    tau_list = np.zeros(len(peaklist))

    # define the windows around each peak. and then use finefit to find the parameters
    for i in range(len(peaklist)):
        curr_pts = [(f > (f[peaklist[i]]-fwindow)) & (f < (f[peaklist[i]]+fwindow))]
        #f0list[i], Qlist[i], Qilist[i], Qclist[i], zclist[i] = finefit(f[curr_pts], z[curr_pts], -30, numspan=2)
        fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i] = finefit(f[curr_pts], z[curr_pts], -30, numspan=2)
        if i==0:
            fit=resfunc3(f[curr_pts], fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
            fitwords = "fr = " + str(fr_list[i]) + "\n" + "Qr = " + str(Qr_list[i]) + "\n" + "Qc = " + str(Qc_list[i]) + "\n" + "a = " + str(a_list[i]) + "\n" + "phi = " + str(phi_list[i]) + "\n" + "tau = " + str(tau_list[i]) + "\n"

            plt.figure(5)
            plt.subplot(2,2,1)
            plt.plot(f[curr_pts], 20*np.log10(np.abs(z[curr_pts])),'.')
            plt.plot(f[curr_pts], 20*np.log10(np.abs(fit)))
            plt.title("resonance " + str(i) + " at " + str(int(10000*fr_list[i])/10000) + " GHz")
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("|S21| [dB]")
            plt.subplot(2,2,2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot(z.real[curr_pts], z.imag[curr_pts],'.')
            plt.plot(fit.real, fit.imag)
            plt.xlabel("S21 real")
            plt.ylabel("S21 imaginary")
            plt.figtext(0.2, 0.2, fitwords)


    # save the lists to fname
    with h5py.File(fname, "r+") as fyle:
        if write == True:
            fyle.__delitem__("{}/f0list".format(chan))
            fyle.__delitem__("{}/zclist".format(chan))
            fyle["{}/f0list".format(chan)] = f0list
            fyle["{}/zclist".format(chan)] = zclist

    plt.show()

    return fr_list, Qr_list, Qc_list
