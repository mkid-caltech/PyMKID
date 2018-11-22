from __future__ import division
from Tkinter import *
import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import math
#import cmath
import matplotlib.pyplot as plt
import h5py
from functools import partial
from matplotlib.backends.backend_pdf import PdfPages

def removecable(f, z, tau, f1):
    """
    returns:
        z_no_cable:  z with the cable delay factor removed (guessing tau, relative to f1?)
    """
    z_no_cable = np.array(z)*np.exp(2j*np.pi*(np.array(f)-f1)*tau)
    return z_no_cable

def estpara(f, z, finit):
    """
    returns:
        f0_est:  The estimated center frequency for this resonance
        Qr_est:  The estimated total quality factor
        id_f0:   The estimated center frequency in index number space
        id_BW:   The 3dB bandwidth in index number space
    """

    realfit = np.polyfit(f,z.real,1)
    imagfit = np.polyfit(f,z.imag,1)
    zfinder = np.sqrt((z.real-(realfit[1]+f*realfit[0]))**2+(z.imag-(imagfit[1]+f*imagfit[0]))**2)
    zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1)+np.append([1,1],zfinder[:-2])+np.append(zfinder[2:],[1,1]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1)+np.append([1,1],zfinder[:-2])+np.append(zfinder[2:],[1,1]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1))/3
    left_trim = np.argmin(zfinder[f<finit])
    right_trim = np.argmin(abs(f-finit)) + np.argmin(zfinder[f>=finit])
    zfinder = zfinder - min(zfinder[left_trim], zfinder[right_trim])
    id_f0 = left_trim + np.argmax(zfinder[left_trim:right_trim+1])
    z0_est = zfinder[id_f0]
    z_3db = z0_est/2
    id_3db_left = left_trim + np.argmin(abs(zfinder[left_trim:id_f0]-z_3db))
    id_3db_right = id_f0 + np.argmin(abs(zfinder[id_f0:right_trim+1]-z_3db))

    #plt.figure()
    #plt.plot(f[left_trim:right_trim+1], abs(z[left_trim:right_trim+1]),'.')
    #plt.plot(f, zfinder, '.')
    #plt.plot(f[left_trim:right_trim+1], zfinder[left_trim:right_trim+1], '.')
    #plt.axvline(x=finit)
    #plt.axvline(x=f[id_f0], color="green")
    #plt.axvline(x=f[id_3db_left], color="red")
    #plt.axvline(x=f[id_3db_right], color="red")
    #plt.axhline(y=0)
    #plt.axhline(y=z_3db, color="red")
    #plt.plot(z[left_trim:right_trim+1].real, z[left_trim:right_trim+1].imag)
    #plt.show()

    f0_est = f[id_f0]
    id_BW = 2*np.mean([abs(id_f0-id_3db_left), abs(id_f0-id_3db_right)])
    Qr_est = f0_est/(2*np.mean([abs(f[id_f0]-f[id_3db_left]), abs(f[id_f0]-f[id_3db_right])]))

    #z_front_avg = np.mean(z[:10])
    #z_back_avg = np.mean(z[-10:])
    #z_avg = np.mean([z_front_avg, z_back_avg])

    #id_f0 = max(np.argmax(abs(np.array(z)-z_avg)),1)

    #z0_est = z[id_f0]
    #f0_est = f[id_f0]
    #z_3db = abs(z_avg - z0_est)/np.sqrt(2)
    #zleft = z[:id_f0]
    #zright = z[id_f0:]

    #id_3db_left = np.argmin(abs(abs(abs(zleft-z_avg)) - z_3db))
    #id_3db_right = np.argmin(abs(abs(abs(zright-z_avg)) - z_3db))
    #id_BW = 2*max(abs(id_f0 - id_3db_left), abs(id_3db_right))
    #Qr_est = f0_est/(id_BW*(f[1]-f[0]))
    #Qr_est = f0_est/(2*max(abs(f[id_f0]-f[id_3db_left]), abs(f[id_3db_right+id_f0]-f[id_f0])))

    return f0_est, Qr_est, id_f0, id_BW

def circle2(z):
    # == METHOD 2b ==
    # "leastsq with jacobian"
    x = z.real
    y = z.imag
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt(((x-xc)**2)+((y-yc)**2))

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri-Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

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

    #t = np.arange(0,2*np.pi,0.002)
    #xcirc = center_2b[0]+r*np.cos(t)
    #ycirc = center_2b[1]+r*np.sin(t)
    #plt.figure()
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.plot(xcirc,ycirc)
    #plt.plot(x, y, 'o')
    #plt.plot(x[int(len(x)/2)], y[int(len(x)/2)],"*")
    #plt.plot(zc.real, zc.imag,"*")
    #plt.plot([zc.real,zc.real+R_2b],[zc.imag,zc.imag])
    #plt.show()

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

def trimdata(f, z, f0, Q, numspan=2):
    f0id = fid(f, f0)
    idspan =  (f0/Q)/((f[-1]-f[0])/(len(f)-1))
    idstart = int(math.floor(max(f0id-(idspan*numspan), 0)))
    idstop  = int(math.floor(min(f0id+(idspan*numspan), len(f))))
    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return fb, zb

def phasefunc(fff, Qff, f0ff, phiff, amp, sign=1):
    phase1 = -phiff+amp*np.arctan(sign*2*Qff*(1-fff/f0ff))
    #phase2 = (phase1+2*np.pi) % (2*np.pi)
    return phase1

def fitphase(f, z, f0g, Qg, numspan=2):
    ang = np.angle(z, deg=False)
    ang = (ang+2*np.pi) % (2*np.pi)
    fstep = (f[-1]-f[0])/(len(f)-1)

    #here#plt.figure()
    #here#plt.axhline(y=2*np.pi, color="black")
    #here#plt.axhline(y=1*np.pi, color="black")
    #here#plt.axhline(y=0, color="black")
    #here#plt.axhline(y=-1*np.pi, color="black")
    #here#plt.axhline(y=-2*np.pi, color="black")
    #plt.plot(f, ang, '.-')

    #for i in range(len(ang)):
    #    if i > 1 and i < (len(ang)-1) and ang[i] > (ang[i-1] + np.pi) and ang[i] > (ang[i+1] + np.pi):
    #        ang[i] = ang[i] - 2*np.pi
    #    elif i > 1 and i < (len(ang)-1) and ang[i] < (ang[i-1] - np.pi) and ang[i] < (ang[i+1] - np.pi):
    #        ang[i] = ang[i] + 2*np.pi

    #here#plt.plot(f,ang,'.-')

    #jumplist = np.array([1], dtype = int)

    #for i in range(len(ang)):
    #    if i > 1 and ang[i] > (ang[i-1] + np.pi):
    #        jumplist = np.append(jumplist, i)
    #        plt.axvline(x=0.5*(f[i]+f[i-1]), color="orange")
            #ang[i:] = ang[i:] - 2*np.pi
    #    elif i > 1 and ang[i] < (ang[i-1] - np.pi):
    #        jumplist = np.append(jumplist, i)
    #        plt.axvline(x=0.5*(f[i]+f[i-1]), color="orange")
            #ang[i:] = ang[i:] + 2*np.pi

    #jumplist = np.append(jumplist, len(ang))

    #if len(jumplist) >=2:
    #    for i in range(len(jumplist)-1):
    #        plt.plot((0.5*(f[jumplist[i]]+f[jumplist[i+1]-1])), np.mean(ang[jumplist[i]:jumplist[i+1]]), 'o', color='black')
    #        print i, (np.mean(ang[jumplist[i]:jumplist[i+1]]) - ang[jumplist[i]-1]),
    #        if (np.mean(ang[jumplist[i]:jumplist[i+1]]) - ang[jumplist[i]-1]) > np.pi:
    #            ang[jumplist[i]:jumplist[i+1]] = ang[jumplist[i]:jumplist[i+1]] - 2*np.pi
    #        elif (np.mean(ang[jumplist[i]:jumplist[i+1]]) - ang[jumplist[i]-1]) < -np.pi:
    #            ang[jumplist[i]:jumplist[i+1]] = ang[jumplist[i]:jumplist[i+1]] + 2*np.pi

    #plt.plot(f,ang,'.-')

    for i in range(len(ang)):
        if i > 0 and ang[i] > (ang[i-1] + np.pi):
            ang[i:] += -2*np.pi
        elif i > 0 and ang[i] < (ang[i-1] - np.pi):
            ang[i:] += +2*np.pi

    while np.mean(ang)>np.pi:
        ang = ang - 2*np.pi
    while np.mean(ang)<-np.pi:
        ang = ang + 2*np.pi

    #here#plt.plot(f, ang, '.-')
    # estimate f0, Q and phi
    #hnumsmopts = int(math.floor(((f0g/Qg)/3)/fstep))
    #hnumsmopts = int(max(min(hnumsmopts, 20), 1))
    #zm = smooth(z, window_len=hnumsmopts*2)
    #angm = unfoldphasenoisy(np.angle(zm, deg=False))      # this filter also does nothing to the current data set
    #dangm = angm[hnumsmopts:] - angm[:-hnumsmopts]
    #fd    =    f[hnumsmopts:] -    f[:-hnumsmopts]
    #idd = np.argmin(dangm)
    #f0 = f[idd + hnumsmopts]

    # linear fit around f0 to find Q and phi
    #idh = idd + hnumsmopts
    #wid = int(math.floor(((f0/Qg)/3)/fstep))
    #fmid = f[max(idh-wid,0):min(idh+wid,len(f))]
    #xmid = (f0-fmid)/f0
    #angmid = ang[max(idh-wid,0):min(idh+wid,len(f))]

    # using polyfit instead of mregress.m
    #cof = np.polyfit(xmid, angmid,1)  # angmid=(cof[0]*xmid)+cof[1]
    #phi = -cof[1];
    #Q = abs(cof[0]/4)

    phi = -np.mean(ang)
    d_height = max(ang)-min(ang)
    tan_max_height = max(2, (d_height/np.pi))

    # set the width of the weighing function to 2df
    #ft, angt = trimdata(f, ang, f0, Q, numspan=numspan)
    #ft = f[id1:id2]
    #angt = ang[id1:id2]
    #ft, zt   = trimdata(f, z, f0, Q, numspan=numspan)

    #plt.show()
    # using robust(?) fit from curve fit
    if np.mean(ang[int(len(ang)/2):]) <= np.mean(ang[:int(len(ang)/2)]):
        fresult = opt.curve_fit(partial(phasefunc, sign=1), f, ang, p0=[Qg, f0g, phi,2], bounds=([0, f[0], -2*np.pi, 0],[10*Qg, f[-1], 2*np.pi, tan_max_height]))
    else:
        fresult = opt.curve_fit(partial(phasefunc, sign=-1), f, ang, p0=[Qg, f0g, phi,2], bounds=([0, f[0], -2*np.pi, 0],[10*Qg, f[-1], 2*np.pi, tan_max_height]))

    #plt.figure()
    #plt.plot(f,ang,'.-')
    #plt.plot(f, phasefunc(f,fresult[0][0],fresult[0][1],fresult[0][2],fresult[0][3]))
    #plt.show()

    return fresult[0], f

def roughfit(f, z, finit, tau0, numspan=2):
    #f1 = f[int(math.floor(len(f)/2))]

    # remove cable term
    z1 = removecable(f, z, tau0, finit)

    # estimate f0, Q (very rough)
    f0_est, Qr_est, id_f0, id_BW = estpara(f,z1, finit)

    # fit circle using data points f0_est +- 2*f0_est/Qr_est
    id1 = max(id_f0-int(id_BW/3), 0)
    id2 = min(id_f0+int(id_BW/3), len(f))

    if len(range(id1, id2)) < 10:
        id1 = 0
        id2 = len(f)

    residue, zc, r = circle2(z1[id1:id2])

    # rotation and traslation to center
    z2 = (zc-z1)*np.exp(-1j*np.angle(zc, deg=False))

    #plt.figure()
    #plt.polar(np.angle(z),np.absolute(z))
    #plt.polar(np.angle(z1),np.absolute(z1))
    #plt.polar(np.angle(z2),np.absolute(z2))
    #plt.show()

    #print Qr_est,

    # trim data and fit phase
    fresult, ft = fitphase(f, z2, f0_est, Qr_est, numspan=numspan)
    #idft1 = fid(f, ft[0])
    #widft = len(ft)
    #zt = z[idft1:idft1+widft] #-1
    #ft = f[idft1:idft1+widft] #-1

    # result
    Q = fresult[0]
    f0 = fresult[1]
    phi = fresult[2]

    zd = -zc/abs(zc)*np.exp(-1j*phi)*r*2
    zinf = zc - zd/2

    Qc = (abs(zinf)/abs(zd))*Q  # Allows negative Qi, but gets best fits somehow
    #Qc = Q*(abs(zc)+r)/(2*r)   # Taken from Gao E.13, but gives negative Qi and doesn't fit as well
    #Qc = Q/(2*r)               # Taken from Gao 4.40, has positive Qi but fits terribly
    Qi = 1/((1/Q)-(1/Qc))
    
    zf0 = zinf + zd

    # projection of zf0 into zinf
    l = np.array(zf0*np.conj(zinf)).real/abs(zinf)**2

    # Q/Qi0 = l
    Qi0 = Q/l

    return f0, Q, phi, zd, zinf, Qc, Qi, Qi0

#def estsig(f, z):
#    # seems to be for estimating variance (sigma squared)
#    sig2 = np.mean(abs(np.diff(z[:int(len(z)/5)]))**2)/2
#    sig2x = sig2/2
#    return sig2x

def resfunc(f, f0, Q, zdx, zdy, zinfx, zinfy, tau, f1=None):
    zd = zdx+1j*zdy
    zinf = zinfx+1j*zinfy
    #Q=Q*10000

    ff = abs(f)
    yy = np.exp(-2j*np.pi*(ff-f1)*tau)*(zinf+(zd/(1+2j*Q*(ff-f0)/f0)))

    yy = np.array(yy)
    ayy = yy.real
    ayy[f>0] = 0
    byy = yy.imag
    byy[f<0] = 0
    y = ayy+byy
    return y

def resfunc2(f, fr, Qr, Qc, areal, aimag, phi0, neg_tau, f1=None):
    x = abs(f)
    a = areal + 1j*aimag
    complexy = a*np.exp(2j*np.pi*(x-f1)*neg_tau)*(1-(((Qr/Qc)*np.exp(1j*phi0))/(1+(2j*Qr*(x-fr)/fr))))

    complexy = np.array(complexy)
    realy = complexy.real
    realy[f>0] = 0
    imagy = complexy.imag
    imagy[f<0] = 0
    y = realy+imagy
    return y

def resfunc3(f, fr, Qr, Qc, a, phi, tau):
    y = a*np.exp(-2j*np.pi*f*tau)*(1-(((Qr/Qc)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return y

def resfunc4(f, fr, Qr, Qc, amag, aarg, phi0, neg_tau):
    x = abs(f)
    a = amag*np.exp(1j*aarg)
    complexy = a*np.exp(2j*np.pi*(x)*neg_tau)*(1-(((Qr/Qc)*np.exp(1j*phi0))/(1+(2j*Qr*(x-fr)/fr))))

    complexy = np.array(complexy)
    realy = complexy.real
    realy[f>0] = 0
    imagy = complexy.imag
    imagy[f<0] = 0
    y = realy+imagy
    return y

def resfunc5(f, fr, Qr, Qc, zinfmag, zinfarg, phi0, tau, Imtau):
    x = abs(f)
    complexy = zinfmag*np.exp(1j*zinfarg)*np.exp(2j*np.pi*(x-fr)*(-tau-1j*Imtau))*(1-(((Qr/Qc)*np.exp(1j*phi0))/(1+(2j*Qr*(x-fr)/fr))))
    complexy = np.array(complexy)
    realy = complexy.real
    realy[f>0] = 0
    imagy = complexy.imag
    imagy[f<0] = 0
    y = realy + imagy
    return y

def finefit(f, z, fr_0, tau_0, fwindow, numspan=2, fit_test=False):
    """
    finefit fits f and z to the resonator model described in Jiansong's thesis
    Input parameters:
        f:        frequencys
        z:        complex impedence
        fr_0:     initially predicted fr
        tau_0:    initially guessed cable delay
        numspan:  how many bandwidths (fr_1/Qr_1) to cut around the data during fit (numspan=1 gives fr_1/Qr_1 on each side for a total width of 2 bandwidth)

    Returns:
        fr_1: frequency center of the resonator
        Q: total Q factor of the resonator
        Qi0: unsure?
            Qi is the internal Q factor of the resonator. Accounts for all the other loss channels
            (Ql, QTL) than through coupling to the feedline (Qc)[paraphrased from Jiansong thesis]
        Qc: coupling Q factor of the resonator. Caused be coupling to the feedline
        zc:
    """
    # find starting parameters using a rough fit
    fr_1, Qr_1, phi_1, zd, zinf, Qc_1, Qi_1, Qi0 = roughfit(f, z, fr_0, tau_0, numspan=numspan)
    print Qi_1,

    #if Q <1000 or Qc_1<3000:
    #    Q = 10000
    #    Qc_1 = 10000

    # estimate variance
    #sig2x = estsig(f,z)

    # trim data
    fnew, znew = trimdata(f, z, fr_1, Qr_1, numspan=numspan)
    if len(fnew)>10:
        f = fnew
        z = znew
        #f1 = f[int(math.floor(len(f)/2))]

    # the fit produced using the starting parameters
    #resctest = 1j*resfunc(f, fr_1, Q, zd.real, zd.imag, zinf.real, zinf.imag, tau_0, f1=f1) + resfunc(-f, fr_1, Q, zd.real, zd.imag, zinf.real, zinf.imag, tau_0, f1=f1)

    # combine x and y data so the fit can go over both simultaneously
    xdata = np.concatenate([-f, f])
    ydata = np.concatenate([z.real, z.imag])

    # separate the real and imaginary parts of zd and zinf
    #zdx = zd.real
    #zdy = zd.imag
    #zinfx = zinf.real
    #zinfy = zinf.imag

    # perform the fine fit
    #fparams, fcov = opt.curve_fit(partial(resfunc, f1=f1), xdata, ydata, p0=[fr_1, Q, zdx, zdy, zinfx, zinfy, tau_0])

    #a_1 = zinf    # a_1=a*np.exp(+2j*np.pi*fr_1*(-tau_0))
    a_1 = zinf*np.exp(-2j*np.pi*fr_1*(-tau_0))
    phi_2 = np.angle(-zd/zinf, deg=False)

    #plt.figure()
    #plt.plot(abs(xdata),ydata,'.')
    #plt.plot(abs(xdata),resfunc5(xdata,fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, 0))
    #plt.show()

    #print max(abs(resfunc2(xdata,fr_1, Qr_1, Qc_1, zinf.real, zinf.imag, phi_2, -tau_0, f1=fr_1)-resfunc4(xdata,fr_1, Qr_1, Qc_1, np.array(a_1*np.exp(-2j*np.pi*fr_1*(-tau_0))).real, np.array(a_1*np.exp(-2j*np.pi*fr_1*(-tau_0))).imag, phi_2, -tau_0)))

    #fparams, fcov = opt.curve_fit(resfunc4, xdata, ydata, p0=[fr_1, Qr_1, Qc_1, abs(a_1), np.angle(a_1*np.exp(-2j*np.pi*fr_1*(-tau_0))), phi_2, -tau_0], bounds=([fr_1-(fwindow/2),0,0,0,-2*np.pi,-2*np.pi,-np.inf],[fr_1+(fwindow/2),+np.inf,+np.inf,+np.inf,+2*np.pi,+2*np.pi,0]))
    #fparams, fcov = opt.curve_fit( partial(resfunc2, f1=fr_1), xdata, ydata, p0=[fr_1, Qr_1, Qc_1, zinf.real, zinf.imag, phi_2, -tau_0], bounds=([fr_1-fwindow,0,0,-np.inf,-np.inf,-2*np.pi,-np.inf],[fr_1+fwindow,+np.inf,+np.inf,+np.inf,+np.inf,2*np.pi,+np.inf]))
    fparams, fcov = opt.curve_fit(resfunc5, xdata, ydata, p0=[fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, 0], bounds=([(fr_1-fwindow), 0, 0, 0, -2*np.pi, -2*np.pi, 0, -20], [(fr_1+fwindow), np.inf, np.inf, np.inf, 2*np.pi, 2*np.pi, np.inf, 20]))
    yexpet = resfunc5(xdata, *fparams)
    chisq = sum(((yexpet - ydata)**2))
    #plt.figure()
    #plt.plot(xdata, ydata, '.')
    #plt.plot(xdata, resfunc2(xdata, fr_1, Q, Qc_1, zinf.real, zinf.imag, phi_2, -tau_0, f1=f1))
    #plt.plot(xdata, resfunc2(xdata, fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5], fparams[6], f1=f1))
    #plt.show()

    # untangle best fit parameters
    #fr_1 = fparams[0]
    #Q = fparams[1]
    #zd = fparams[2]+1j*fparams[3]
    #zinf = fparams[4]+1j*fparams[5]
    #tau = fparams[6]

    fr_fine = fparams[0]
    Qr_fine = fparams[1]
    Qc_fine = fparams[2]
    #a_fine = (fparams[3] + 1j*fparams[4])*np.exp(-2j*np.pi*fr_1*fparams[6])
    a_fine = fparams[3]*np.exp(1j*fparams[4])*np.exp(2j*np.pi*fparams[0]*(fparams[6]+1j*fparams[7]))
    #a_fine = fparams[3]*np.exp(1j*fparams[4])
    phi_fine = fparams[5]
    tau_fine = fparams[6] + 1j*fparams[7]

    #print Qr_1, Qr_fine

    # find Qc, unsure where this equation originates
    #Qc = (abs(zinf)/abs(zd))*Q
    # find Qi (all Q that isn't Qc)
    Qi_fine = 1/((1/Qr_fine)-(1/Qc_fine))
    print Qi_fine


    # the next section finds Qi0 and other parameters. I'm not sure where any of these equations come from or why we want these parameters
    zf0 = zinf + zd
    # projection of zf0 into zinf
    l = (zf0*np.conj(zinf)).real/abs(zinf)**2
    # Q/Qi0 = l
    Qi0 = Qr_1/l
    zc = zinf + zd/2
    r = abs(zd/2)
    phi = np.angle(-zd/zc, deg=False)

    # plot the rough and fine fits to compare
    #plt.figure(4)

    # the fit produced using the best parameters
    #resftest = 1j*resfunc2(f, fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5], fparams[6], f1=f1) + resfunc2(-f, fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5], fparams[6], f1=f1)
    #resftest = resfunc3(f, fr_fine, Qr_fine, Qc_fine, a_fine, phi_fine, tau_fine)

    # plot rough fit in red
    #plt.plot(f, 20*np.log10(np.abs(resctest)), 'r')

    # plot fine fit in blue
    #plt.plot(f, 20*np.log10(np.abs(resftest)), 'b')

    # plot real data
    #plt.plot(f, 20*np.log10(np.abs(z)), '.')

    #return fr_1, Q, Qi0, Qc, zc
    if fit_test:
        return fr_fine, Qr_fine, Qc_fine, a_fine, phi_fine, tau_fine, chisq
    else:
        return fr_fine, Qr_fine, Qc_fine, a_fine, phi_fine, tau_fine

def sweep_fit(fname, nsig=3, fwindow=5e-4, chan="S21", rewrite=False, freqfile=False, additions=[]):
    """
    sweep_fit fits data taken using save_scatter to the resonator model described in Jiansong's thesis
    Input parameters:
        fname: name of the file containing save_scatter data. eg "YY160622.h5"
        nsig: number of sigma above which a peak is considered a resonator peak
        fwindow: half width of the frequency space window cut around each resonator before the resonator is fit
        chan: channel from save_scatter being analyzed
        rewrite: whether or not sweep_fit will save its fit data to the fname file and pdf
        freqfile: whether or not sweep_fit will make a file with resonance locations (for Templar)

    Returns:
        fr_list, Qr_list, Qc_list (lists of values for each resonator) save to the fname file
    """
    # open file and read data
    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])

    nfreq = 1/(2*((f[-1]-f[0])/(len(f)-1)))    # The nyquist frequency [s]
    evfreq = 1/(2*fwindow)    # The frequency corresponding to the expected window size [s]
    b, a = sig.butter(2, evfreq/nfreq, btype='highpass')
    mfz = np.sqrt(sig.filtfilt(b, a, z.real)**2 + sig.filtfilt(b, a, z.imag)**2)  # The magnitude of filtered z

    # Do some averaging
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0)+np.append([0,0],mfz[:-2])+np.append(mfz[2:],[0,0]))/5
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0))/3

    # Record the standard deviation of mfz
    bstd = np.std(mfz)

    # initialize peaklist
    peaklist = np.array([], dtype = int)

    # add the manually entered frequencies to peaklist
    for added in additions:
        peaklist = np.append(peaklist, np.argmin(abs(f-added)))
    addlist = peaklist

    # initialize mn above max and mx below min
    #mn = +np.inf
    mx = -np.inf
    peak_pos = 0
    #mn_pos = np.nan
    mx_pos = np.nan
    lookformax = False
    delta = nsig*bstd
    gamma = 3*np.mean(mfz[mfz<delta])

    # find peaks and add them to peaklist
    for i in range(len(mfz)):
        cp = mfz[i]
        if cp >= mx:
            mx = cp
            mx_pos = i
    #    if cp <= mn:
    #        mn = cp
            #mn_pos = i
        if lookformax == True:
            if cp < gamma:
                peak_pos = mx_pos
                peaklist = np.append(peaklist, peak_pos)
    #            mn = cp
                #mn_pos = i
                lookformax = False
        else:
            if cp > delta and f[i] > (f[peak_pos]+2*fwindow):
                mx = cp
                mx_pos = i
                lookformax = True

    peaklist = sorted(peaklist)

    # Plot the transmission and peaks
    plt.figure(figsize=(10, 10))
    fig, axarr = plt.subplots(nrows=2, sharex=True, num=1)
    axarr[0].plot(f, 20*np.log10(np.abs(np.array(z)))) # plot the unaltered transmission
    axarr[0].set_title('Transmission with Resonance Identification')
    axarr[0].set_ylabel("|$S_{21}$| [dB]")
    axarr[1].plot(f, mfz/bstd)
    axarr[1].set_ylabel("|filtered z| [#std]")
    axarr[1].set_xlabel("Frequency [GHz]")
    #curr_xlims = [min(f), max(f)]
    #nsigline =  bstd*nsig*np.array([1, 1])
    #axarr[1].plot(curr_xlims, nsigline, 'r-')
    axarr[1].axhline(y=nsig, color="red", label="nsig = "+str(nsig))
    axarr[1].axhline(y=gamma/bstd, color="green")
    axarr[1].plot(f[peaklist], mfz[peaklist]/bstd, 'gs', label=str(len(peaklist)-len(additions))+" resonances identified")
    axarr[1].plot(f[addlist], mfz[addlist]/bstd, 'ys', label=str(len(addlist))+" resonances manually added")
    plt.legend()
    # Save to pdf if rewrite == True
    if rewrite == True:
        Res_pdf = PdfPages(fname[:-3]+'.pdf')
        Res_pdf.savefig()

    plt.show()

    # initialize the parameter lists
    fr_list = np.zeros(len(peaklist))
    Qr_list = np.zeros(len(peaklist))
    Qc_list = np.zeros(len(peaklist))
    a_list = np.array([0.+0j]*len(peaklist))
    phi_list = np.zeros(len(peaklist))
    tau_list = np.array([0.+0j]*len(peaklist))

    # define the windows around each peak. and then use finefit to find the parameters
    for i in range(len(peaklist)):
        curr_pts = [(f >= (f[peaklist[i]]-fwindow)) & (f <= (f[peaklist[i]]+fwindow))]
        f_curr = f[curr_pts]
        z_curr = z[curr_pts]
        print 'Resonance #{}'.format(str(i)),
        fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i] = finefit(f_curr, z_curr, f[peaklist[i]], 30, fwindow, numspan=2)

        if rewrite == True:
            fit = resfunc3(f_curr, fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
            zrfit = resfunc3(fr_list[i], fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
            fit_down = resfunc3(f_curr, fr_list[i], 0.95*Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
            fit_up = resfunc3(f_curr, fr_list[i], 1.05*Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
            fitwords = "$f_{r}$ = " + str(fr_list[i]) + "\n" + "$Q_{r}$ = " + str(Qr_list[i]) + "\n" + "$Q_{c}$ = " + str(Qc_list[i]) + "\n" + "$Q_{i}$ = " + str((Qr_list[i]*Qc_list[i])/(Qc_list[i]-Qr_list[i])) + "\n" + "$a$ = " + str(a_list[i]) + "\n" + "$\phi_{0}$ = " + str(phi_list[i]) + "\n" + r"$\tau$ = " + str(tau_list[i]) + "\n"

            plt.figure(figsize=(10, 10))

            plt.subplot(2,2,1)
            plt.plot(f_curr, 20*np.log10(np.abs(z_curr)),'.', label='Data')
            plt.plot(f_curr, 20*np.log10(np.abs(fit_down)), label='Fit 0.95Q')
            plt.plot(f_curr, 20*np.log10(np.abs(fit)), label='Fit 1.00Q')
            plt.plot(f_curr, 20*np.log10(np.abs(fit_up)), label='Fit 1.05Q')
            plt.plot(fr_list[i], 20*np.log10(np.abs(zrfit)), '*', markersize=10, color='red', label='$f_{r}$')
            plt.title("resonance " + str(i) + " at " + str(int(10000*fr_list[i])/10000) + " GHz")
            plt.xlabel("Frequency [GHz]")
            plt.xticks([min(f_curr),max(f_curr)])
            plt.ylabel("|$S_{21}$| [dB]")
            plt.legend(bbox_to_anchor=(2, -0.15))

            plt.subplot(2,2,2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot(z_curr.real, z_curr.imag,'.', label='Data')
            plt.plot(fit_down.real, fit_down.imag,  label='Fit 0.95Q')
            plt.plot(fit.real, fit.imag,  label='Fit 1.00Q')
            plt.plot(fit_up.real, fit_up.imag, label='Fit 1.05Q')
            plt.plot(zrfit.real, zrfit.imag, '*', markersize=10, color='red',  label='Fr')
            plt.xlabel("$S_{21}$ real")
            plt.xticks([min(z_curr.real),max(z_curr.real)])
            plt.ylabel("$S_{21}$ imaginary")
            plt.yticks([min(z_curr.imag),max(z_curr.imag)])

            plt.figtext(0.6, 0.085, fitwords)
            plt.figtext(0.55, 0.26, r"$S_{21}(f)=ae^{-2\pi jf\tau}\left [ 1-\frac{Q_{r}/Q_{c}e^{j\phi_{0}}}{1+2jQ_{r}(\frac{f-f_{r}}{f_{r}})} \right ]$", fontsize=20)

            plt.subplot(2,2,3, projection="polar")
            zi_no_cable = removecable(f_curr, z_curr, tau_list[i], 0)/(a_list[i])
            zi_normalized = 1-((1 - zi_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zi_normalized), np.absolute(zi_normalized),'.')
            zfit_no_cable = removecable(f_curr, fit, tau_list[i], 0)/(a_list[i])
            zfit_normalized = 1-((1 - zfit_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zfit_normalized), np.absolute(zfit_normalized), color='red')
            zrfit_no_cable = removecable(fr_list[i], zrfit, tau_list[i], 0)/(a_list[i])
            zrfit_normalized = 1-((1 - zrfit_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zrfit_normalized), np.absolute(zrfit_normalized),'*', markersize=10, color='red')

            Res_pdf.savefig()
            plt.close()

    if rewrite == True:
        Res_pdf.close()

    # save the lists to fname
    with h5py.File(fname, "r+") as fyle:
        if rewrite == True:
            if "{}/fr_list".format(chan) in fyle:
                fyle.__delitem__("{}/fr_list".format(chan))
            if "{}/Qr_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qr_list".format(chan))
            fyle["{}/fr_list".format(chan)] = fr_list
            fyle["{}/Qr_list".format(chan)] = Qr_list

    if freqfile == True:
        freqfile_data = np.transpose(np.array([np.append(1,fr_list), np.zeros(len(fr_list)+1), np.zeros(len(fr_list)+1), np.append(0,-20*np.ones(len(fr_list)))]))
        np.savetxt("freqfile.txt", freqfile_data, fmt='%.10f')

    return fr_list, Qr_list, Qc_list

def sweep_test(fname, nsig=3, fwindow=5e-4, chan="S21", additions=[]):
    # open file and read data
    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])

    nfreq = 1/(2*((f[-1]-f[0])/(len(f)-1)))    # The nyquist frequency [s]
    evfreq = 1/(2*fwindow)    # The frequency corresponding to the expected window size [s]
    b, a = sig.butter(2, evfreq/nfreq, btype='highpass')    # The forward-backward filter parameters

    mfz = np.sqrt(sig.filtfilt(b, a, z.real)**2 + sig.filtfilt(b, a, z.imag)**2)    # The magnitude of independently filtered z

    # Do some averaging...
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0)+np.append([0,0],mfz[:-2])+np.append(mfz[2:],[0,0]))/5
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0))/3

    # Record the standard deviation of mfz
    bstd = np.std(mfz)

    # Initialize the peaklist
    peaklist = np.array([], dtype = int)

    # add the manually entered frequencies to peaklist
    for added in additions:
        peaklist = np.append(peaklist, np.argmin(abs(f-added)))
    addlist = peaklist

    # Initialize mx below min
    mx = -np.inf
    peak_pos = 0
    mx_pos = np.nan
    lookformax = False
    delta = nsig*bstd
    gamma = 3*np.mean(mfz[mfz<delta])

    # Find peaks and add them to peaklist
    for i in range(len(mfz)):
        cp = mfz[i]
        if cp >= mx:
            mx = cp
            mx_pos = i
        if lookformax == True:
            if cp < gamma:
                peak_pos = mx_pos
                peaklist = np.append(peaklist, peak_pos)
                lookformax = False
        else:
            if cp > delta and f[i] > (f[peak_pos]+2*fwindow):
                mx = cp
                mx_pos = i
                lookformax = True

    peaklist = sorted(peaklist)

    # Plot the transmission and peaks
    plt.figure(figsize=(10, 10))
    fig, axarr = plt.subplots(nrows=2, sharex=True, num=1)
    axarr[0].plot(f, 20*np.log10(np.abs(np.array(z))))    # Plot the unaltered transmission
    axarr[0].set_title('Transmission with Resonance Identification')
    axarr[0].set_ylabel("|$S_{21}$| [dB]")
    axarr[1].plot(f, mfz/bstd)
    axarr[1].set_ylabel("|filtered z| [#std]")
    axarr[1].set_xlabel("Frequency [GHz]")
    axarr[1].axhline(y=nsig, color="red")
    axarr[1].axhline(y=gamma/bstd, color="green")
    axarr[1].plot(f[peaklist], mfz[peaklist]/bstd, 'gs', label=str(len(peaklist)-len(additions))+" resonances identified")
    axarr[1].plot(f[addlist], mfz[addlist]/bstd, 'ys', label=str(len(addlist))+" resonances manually added")
    plt.legend()
    plt.show()


def sweep_fit2(fname, nsig=3, fwindow=5e-4, chan="S21", rewrite=False, freqfile=False, additions=[], KIDs=80):
    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])
    
    nfreq = 1/(2*((f[-1]-f[0])/(len(f)-1)))    # The nyquist frequency [s]
    evfreq = 1/(2*fwindow)    # The frequency corresponding to the expected window size [s]
    b, a = sig.butter(2, evfreq/nfreq, btype='highpass')
    mfz0 = np.sqrt(sig.filtfilt(b, a, z.real)**2 + sig.filtfilt(b, a, z.imag)**2)  # The magnitude of filtered z
        
    # Do some averaging
    mfz = (mfz0+np.append(0,mfz0[:-1])+np.append(mfz0[1:],0)+np.append([0,0],mfz0[:-2])+np.append(mfz0[2:],[0,0]))/5
    mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0))/3
    for pie in range(KIDs):

        
        # Record the standard deviation of mfz
        bstd = np.std(mfz)
        
        # initialize peaklist
        peaklist = np.array([], dtype = int)
        
        # add the manually entered frequencies to peaklist
        for added in additions:
            peaklist = np.append(peaklist, np.argmin(abs(f-added)))
        addlist = peaklist

        peaklist = [np.argmax(mfz)]

        if True == True:
            plt.figure(1)
            plt.plot(f, mfz)
            plt.plot(f[peaklist], mfz[peaklist], 'gs')
            plt.show()
        
        # initialize the parameter lists
        fr_list = np.zeros(len(peaklist))
        Qr_list = np.zeros(len(peaklist))
        Qc_list = np.zeros(len(peaklist))
        a_list = np.array([0.+0j]*len(peaklist))
        phi_list = np.zeros(len(peaklist))
        tau_list = np.array([0.+0j]*len(peaklist))
        
        # define the windows around each peak. and then use finefit to find the parameters
        for i in range(len(peaklist)):
            curr_pts = [(f >= (f[peaklist[i]]-fwindow)) & (f <= (f[peaklist[i]]+fwindow))]
            f_curr = f[curr_pts]
            z_curr = z[curr_pts]
            print pie,
            fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i] = finefit(f_curr, z_curr, f[peaklist[i]], 30, fwindow, numspan=2)

        fit = resfunc3(f_curr, fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
        zrfit = resfunc3(fr_list[i], fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
        fit_down = resfunc3(f_curr, fr_list[i], 0.95*Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
        fit_up = resfunc3(f_curr, fr_list[i], 1.05*Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i])
        fitwords = "$f_{r}$ = " + str(fr_list[i]) + "\n" + "$Q_{r}$ = " + str(Qr_list[i]) + "\n" + "$Q_{c}$ = " + str(Qc_list[i]) + "\n" + "$Q_{i}$ = " + str((Qr_list[i]*Qc_list[i])/(Qc_list[i]-Qr_list[i])) + "\n" + "$a$ = " + str(a_list[i]) + "\n" + "$\phi_{0}$ = " + str(phi_list[i]) + "\n" + r"$\tau$ = " + str(tau_list[i]) + "\n"
        
        if True == False:
            plt.figure(2, figsize=(10, 10))

            plt.subplot(2,2,1)
            plt.plot(f_curr, 20*np.log10(np.abs(z_curr)),'.', label='Data')
            plt.plot(f_curr, 20*np.log10(np.abs(fit_down)), label='Fit 0.95Q')
            plt.plot(f_curr, 20*np.log10(np.abs(fit)), label='Fit 1.00Q')
            plt.plot(f_curr, 20*np.log10(np.abs(fit_up)), label='Fit 1.05Q')
            plt.plot(fr_list[i], 20*np.log10(np.abs(zrfit)), '*', markersize=10, color='red', label='$f_{r}$')
            plt.title("resonance " + str(pie) + " at " + str(int(10000*fr_list[i])/10000) + " GHz")
            plt.xlabel("Frequency [GHz]")
            plt.xticks([min(f_curr),max(f_curr)])
            plt.ylabel("|$S_{21}$| [dB]")
            plt.legend(bbox_to_anchor=(2, -0.15))

            plt.subplot(2,2,2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot(z_curr.real, z_curr.imag,'.', label='Data')
            plt.plot(fit_down.real, fit_down.imag,  label='Fit 0.95Q')
            plt.plot(fit.real, fit.imag,  label='Fit 1.00Q')
            plt.plot(fit_up.real, fit_up.imag, label='Fit 1.05Q')
            plt.plot(zrfit.real, zrfit.imag, '*', markersize=10, color='red',  label='Fr')
            plt.xlabel("$S_{21}$ real")
            plt.xticks([min(z_curr.real),max(z_curr.real)])
            plt.ylabel("$S_{21}$ imaginary")
            plt.yticks([min(z_curr.imag),max(z_curr.imag)])
            
            plt.figtext(0.6, 0.085, fitwords)
            plt.figtext(0.55, 0.26, r"$S_{21}(f)=ae^{-2\pi jf\tau}\left [ 1-\frac{Q_{r}/Q_{c}e^{j\phi_{0}}}{1+2jQ_{r}(\frac{f-f_{r}}{f_{r}})} \right ]$", fontsize=20)
            
            plt.subplot(2,2,3, projection="polar")
            zi_no_cable = removecable(f_curr, z_curr, tau_list[i], 0)/(a_list[i])
            zi_normalized = 1-((1 - zi_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zi_normalized), np.absolute(zi_normalized),'.')
            zfit_no_cable = removecable(f_curr, fit, tau_list[i], 0)/(a_list[i])
            zfit_normalized = 1-((1 - zfit_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zfit_normalized), np.absolute(zfit_normalized), color='red')
            zrfit_no_cable = removecable(fr_list[i], zrfit, tau_list[i], 0)/(a_list[i])
            zrfit_normalized = 1-((1 - zrfit_no_cable)/np.exp(1j*(phi_list[i])))
            plt.plot(np.angle(zrfit_normalized), np.absolute(zrfit_normalized),'*', markersize=10, color='red')

            plt.show()

        curr_pts_ind = np.array(range(len(f)))
        #curr_pts_ind = curr_pts_ind[curr_pts]
        bandwidths = 20
        justthis = resfunc3(f, fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i].real)*np.exp(2*np.pi*fr_list[i]*tau_list[i].imag)
        print 'final', resfunc3(f, fr_list[i], Qr_list[i], Qc_list[i], a_list[i], phi_list[i], tau_list[i].real)
        print tau_list[i].imag
        print np.exp(2*np.pi*fr_list[i]*tau_list[i].imag)
        print justthis
        
        if True == False:
            plt.figure(3)
            plt.plot(f, 20*np.log10(np.abs(np.array(z))))
            plt.plot(f, 20*np.log10(np.abs(np.array(justthis))))
            plt.show()


        mfz0 = mfz0 - np.sqrt(sig.filtfilt(b, a, justthis.real)**2 + sig.filtfilt(b, a, justthis.imag)**2)    # The magnitude of independently filtered z
        mfz0[mfz0<0] = 0

        # Do some averaging...
        mfz = (mfz0+np.append(0,mfz0[:-1])+np.append(mfz0[1:],0)+np.append([0,0],mfz0[:-2])+np.append(mfz0[2:],[0,0]))/5
        mfz = (mfz+np.append(0,mfz[:-1])+np.append(mfz[1:],0))/3

#z = np.delete(z,curr_pts_ind[np.bitwise_and(f>(fr_list[i]*(1-(bandwidths/(2*Qr_list[i])))), f<(fr_list[i]*(1+(bandwidths/(2*Qr_list[i])))))])
#       mfz = np.delete(mfz,curr_pts_ind[np.bitwise_and(f>(fr_list[i]*(1-(bandwidths/(2*Qr_list[i])))), f<(fr_list[i]*(1+(bandwidths/(2*Qr_list[i])))))])
#       f = np.delete(f,curr_pts_ind[np.bitwise_and(f>(fr_list[i]*(1-(bandwidths/(2*Qr_list[i])))), f<(fr_list[i]*(1+(bandwidths/(2*Qr_list[i])))))])
#bandwidths = 20
#       plt.plot(f[np.bitwise_and(f>(fr_list[i]*(1-(bandwidths/(2*Qr_list[i])))), f<(fr_list[i]*(1+(bandwidths/(2*Qr_list[i])))))], 20*np.log10(np.abs(np.array(z[np.bitwise_and(f>(fr_list[i]*(1-(bandwidths/(2*Qr_list[i])))), f<(fr_list[i]*(1+(bandwidths/(2*Qr_list[i])))))]))))

