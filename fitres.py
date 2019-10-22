from __future__ import division
import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import math
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

    if False:
        plt.figure()
        plt.plot(f[left_trim:right_trim+1], abs(z[left_trim:right_trim+1]),'.')
        plt.plot(f, zfinder, '.')
        plt.plot(f[left_trim:right_trim+1], zfinder[left_trim:right_trim+1], '.')
        plt.axvline(x=finit)
        plt.axvline(x=f[id_f0], color="green")
        plt.axvline(x=f[id_3db_left], color="red")
        plt.axvline(x=f[id_3db_right], color="red")
        plt.axhline(y=0)
        plt.axhline(y=z_3db, color="red")
        plt.show()

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

    t = np.arange(0,2*np.pi,0.002)
    xcirc = center_2b[0]+r*np.cos(t)
    ycirc = center_2b[1]+r*np.sin(t)

    if False:
        plt.figure()
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(xcirc,ycirc)
        plt.plot(x, y, 'o')
        plt.plot(x[int(len(x)/2)], y[int(len(x)/2)],"*")
        plt.plot(zc.real, zc.imag,"*")
        plt.plot([zc.real,zc.real+R_2b],[zc.imag,zc.imag])
        plt.show()

    return residue, zc, r

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

def phasefunc(f, Qr, fr, phi, amplitude, sign=1):
    # Jiansong's equation 4.42 with theta = arg(z) + phi from figure E.3
    argz = -phi-amplitude*np.arctan(sign*2*Qr*((f-fr)/fr))
    return argz

def fitphase(f, z, f0g, Qg, numspan=2, resnum=None, plotting=True):
    argz = np.angle(z, deg=False)
    fstep = (f[-1]-f[0])/(len(f)-1)

    if False:
        plt.figure(3)
        plt.axhline(y=2*np.pi, color="black")
        plt.axhline(y=1*np.pi, color="0.25")
        plt.axhline(y=0, color="0.75")
        plt.axhline(y=-1*np.pi, color="0.25")
        plt.axhline(y=-2*np.pi, color="black")
        plt.plot(f,argz,'.-', label='raw')

    for i in range(len(argz)):
        if i > 1 and abs(argz[i]-argz[i-1]) > (0.5*np.pi):
            fresult = opt.curve_fit(partial(phasefunc, sign=1),f[:i],argz[:i],p0=[Qg, f0g, np.pi,2],bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, max(2, ((max(argz)-min(argz))/np.pi))]))
            guide = phasefunc(f,fresult[0][0],fresult[0][1],fresult[0][2],fresult[0][3])
            if plotting:
                plt.plot(f, phasefunc(f, Qg, f0g, np.pi, 2), label='guess')
                plt.plot(f, guide, label='adjuster fit')
            argz[i:] += -2*np.pi*round((argz[i]-guide[i+1])/(2*np.pi))

    if False:
        plt.plot(f, argz, '.-', label='adjusted')

    while np.mean(argz)>np.pi:
        argz = argz - 2*np.pi
    while np.mean(argz)<-np.pi:
        argz = argz + 2*np.pi

    if False:
        plt.plot(f, argz, '.-', label='adjusted+shifted')

    phi0 = -np.mean(argz)
    d_height = max(argz)-min(argz)
    tan_max_height = max(2, (d_height/np.pi))

    # using robust(?) fit from curve fit
    if np.mean(argz[int(len(argz)/2):]) <= np.mean(argz[:int(len(argz)/2)]):
        fresult = opt.curve_fit(partial(phasefunc, sign=1), f, argz, p0=[Qg, f0g, phi0, 2], bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, tan_max_height]))
    else:
        fresult = opt.curve_fit(partial(phasefunc, sign=-1), f, argz, p0=[Qg, f0g, phi0, 2], bounds=([0, min(f[0],f[-1]), -2*np.pi, 0],[10*Qg, max(f[0],f[-1]), 2*np.pi, tan_max_height]))

    if False:
        plt.plot(f, phasefunc(f,fresult[0][0],fresult[0][1],fresult[0][2],fresult[0][3],sign=1), '-', label='fit')
        plt.legend()
        plt.show()

    return fresult[0], f

def roughfit(f, z, finit, tau0, numspan=2, resnum=None, plotting=True):

    # remove cable term
    if False:
        f0_est0, Qr_est0, id_f00, id_BW0 = estpara(f,z, finit)
        z_edges = np.append(z[:int(id_f00-2*id_BW0)],z[int(id_f00+2*id_BW0):])
        f_edges = np.append(f[:int(id_f00-2*id_BW0)],f[int(id_f00+2*id_BW0):])
        edge_angle_fit = np.polyfit(f_edges,np.angle(z_edges),1)
        edge_angle_line = edge_angle_fit[1]+f_edges*edge_angle_fit[0]
        plt.figure(1)
        plt.ylabel('Im(z)')
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag,'.')
        plt.plot(z[:int(id_f00-2*id_BW0)].real,z[:int(id_f00-2*id_BW0)].imag,'.')
        plt.plot(z[int(id_f00+2*id_BW0):].real,z[int(id_f00+2*id_BW0):].imag,'.')
        plt.figure(2)
        plt.ylabel('abs(z)')
        plt.plot(f,abs(z))
        plt.plot(f[:int(id_f00-2*id_BW0)],abs(z[:int(id_f00-2*id_BW0)]))
        plt.plot(f[int(id_f00+2*id_BW0):],abs(z[int(id_f00+2*id_BW0):]))
        plt.figure(3)
        plt.ylabel('angle(z)')
        plt.plot(f,np.angle(z))
        plt.plot(f[:int(id_f00-2*id_BW0)],np.angle(z[:int(id_f00-2*id_BW0)]))
        plt.plot(f[int(id_f00+2*id_BW0):],np.angle(z[int(id_f00+2*id_BW0):]))
        plt.plot(f_edges,edge_angle_line,label=edge_angle_fit[0]/(-2*np.pi))
        plt.legend()
        plt.show()
    z1 = removecable(f, z, tau0, finit)

    # estimate f0 (pretty good), Q (very rough)
    f0_est, Qr_est, id_f0, id_BW = estpara(f,z1, finit)

    # fit circle using trimmed data points
    id1 = max(id_f0-int(id_BW/3), 0)
    id2 = min(id_f0+int(id_BW/3), len(f))
    if len(range(id1, id2)) < 10:
        id1 = 0
        id2 = len(f)

    residue, zc, r = circle2(z1[id1:id2])
    #print '|a| ', abs(zc)+r, 20*np.log10(abs(zc)+r), ' dB'

    # rotation and traslation to center
    z1b = z1*np.exp(-1j*np.angle(zc, deg=False))
    #z1b = (z1-zc)*np.exp(-1j*np.angle((z1-zc)[id_f0], deg=False)+1j*np.pi)
    z2 = (zc-z1)*np.exp(-1j*np.angle(zc, deg=False))

    if False:
        plt.figure(2)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag,'o', color=None, label='z')
        plt.plot(z1.real,z1.imag,'o', color=None, label='z1')
        #plt.plot(z1b.real,z1b.imag,'o', color=None, label='z1b')
        #plt.plot(z2.real,z2.imag,'o', color=None, label='z2')
        #plt.plot(z2[np.argmin(abs(f-f0_est))].real,z2[np.argmin(abs(f-f0_est))].imag,'o', color='orange', label='f0 est')
        plt.legend()
        plt.show()

    # trim data and fit phase
    fresult, ft = fitphase(f, z2, f0_est, Qr_est, numspan=numspan, resnum=resnum, plotting=plotting)
    #idft1 = fid(f, ft[0])
    #widft = len(ft)
    #zt = z[idft1:idft1+widft] #-1
    #ft = f[idft1:idft1+widft] #-1

    # result
    Q = fresult[0]
    f0 = fresult[1]
    phi = fresult[2]
    #print 'alleged phi is '+str(phi)

    zd = -zc/abs(zc)*np.exp(-1j*(phi))*r*2
    zinf = zc - zd/2
    #print np.angle(-zd/zinf, deg=False), np.angle(np.exp(-1j*(phi))+r/abs(zc), deg=False), phi-np.angle(zc, deg=False)

    Qc = Q*(abs(zinf)/(2*r))  # Allows negative Qi, but gets best fits somehow
    #Qc = Q*(abs(zc)+r)/(2*r)   # Taken from Gao E.13, but gives negative Qi and doesn't fit as well
    #Qc = Q/(2*r)               # Taken from Gao 4.40, has positive Qi but fits terribly
    Qi = 1/((1/Q)-(1/Qc))

    zf0 = zinf + zd

    # projection of zf0 into zinf
    l = np.array(zf0*np.conj(zinf)).real/abs(zinf)**2

    # Q/Qi0 = l
    Qi0 = Q/l

    if False:
        x_lin_reg = (f-f[id_f0])/f[id_f0]
        lin_reg_0 = max(id_f0-int(id_BW/20), 0)
        lin_reg_1 = min(id_f0+int(id_BW/20), len(f))
        lin_reg_params = np.polyfit(x_lin_reg[lin_reg_0:lin_reg_1],abs(z1[lin_reg_0:lin_reg_1]),1)
        lin_reg_fit = lin_reg_params[1]+x_lin_reg[lin_reg_0:lin_reg_1]*lin_reg_params[0]
        Lambda = lin_reg_params[0]*lin_reg_params[1]
        Qm_000 = -2*(abs(zc)+r)*(abs(zc)+r)*Q*Q/Lambda
        Qc_000 = Q/(1-Q*np.sqrt(abs(-2*(lin_reg_params[1]/(lin_reg_params[0]*Qm_000))-1/(Qm_000**2))))
        all_fit = (abs(zc)+r)*np.sqrt(((1-(((Q/Qc_000)+2*Q*Q*x_lin_reg/Qm_000)/(1+4*Q*Q*x_lin_reg*x_lin_reg)))**2)+((((Q/Qm_000)-2*Q*Q*x_lin_reg/Qc_000)/(1+4*Q*Q*x_lin_reg*x_lin_reg))**2))
        plt.plot(x_lin_reg,abs(z1))
        plt.plot(x_lin_reg[lin_reg_0:lin_reg_1],abs(z1[lin_reg_0:lin_reg_1]))
        plt.plot(x_lin_reg[id_f0],abs(z1[id_f0]),'o')
        plt.plot(x_lin_reg[lin_reg_0:lin_reg_1],lin_reg_fit, label=Qm_000)
        plt.plot(x_lin_reg,all_fit)
        plt.legend()
        plt.show()
        print Qm_000, Qc_000
        print abs(zc)-r, lin_reg_params[1]
    Qm_000=10

    return f0, Q, phi, zd, zinf, Qc, Qi, Qi0, Qm_000

#def estsig(f, z):
#    # seems to be for estimating variance (sigma squared)
#    sig2 = np.mean(abs(np.diff(z[:int(len(z)/5)]))**2)/2
#    sig2x = sig2/2
#    return sig2x

#def resfunc0(f, f0, Q, zdx, zdy, zinfx, zinfy, tau, f1=None):
#    zd = zdx+1j*zdy
#    zinf = zinfx+1j*zinfy
#    #Q=Q*10000
#
#    ff = abs(f)
#    yy = np.exp(-2j*np.pi*(ff-f1)*tau)*(zinf+(zd/(1+2j*Q*(ff-f0)/f0)))
#
#    yy = np.array(yy)
#    ayy = yy.real
#    ayy[f>0] = 0
#    byy = yy.imag
#    byy[f<0] = 0
#    y = ayy+byy
#    return y

#def resfunc2(f, fr, Qr, Qc, areal, aimag, phi0, neg_tau, f1=None):
#    x = abs(f)
#    a = areal + 1j*aimag
#    complexy = a*np.exp(2j*np.pi*(x-f1)*neg_tau)*(1-(((Qr/Qc)*np.exp(1j*phi0))/(1+(2j*Qr*(x-fr)/fr))))
#
#    complexy = np.array(complexy)
#    realy = complexy.real
#    realy[f>0] = 0
#    imagy = complexy.imag
#    imagy[f<0] = 0
#    y = realy+imagy
#    return y

def resfunc3(f, fr, Qr, Qc_hat_mag, a, phi, tau):
    """A semi-obvious form of Gao's S21 function. e^(2j*pi*fr*tau) is incorporated into a."""
    S21 = a*np.exp(-2j*np.pi*(f-fr)*tau)*(1-(((Qr/Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return S21

def resfunc5(f_proj, fr, Qr,  Qc_hat_mag, zinfmag, zinfarg, phi, tau, Imtau):
    """An alternate resfunc3 with all real inputs. Re(S21) is projected onto negative frequency space."""
    f = abs(f_proj)
    #a = zinfmag*np.exp(1j*zinfarg)*np.exp(-2j*np.pi*fr*(tau+1j*Imtau))
    S21 = zinfmag*np.exp(1j*zinfarg)*np.exp(-2j*np.pi*(f-fr)*(tau+1j*Imtau))*(1-(((Qr/ Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    S21 = np.array(S21)
    real_S21 = S21.real
    real_S21[f_proj>0] = 0
    imag_S21 = S21.imag
    imag_S21[f_proj<0] = 0
    return real_S21 + imag_S21

def resfunc7(f_proj, fr, Qr, Qc, zinfmag, zinfarg, Qm, tau, Imtau):
    """An alternate resfunc3 with all real inputs. Re(S21) is projected onto negative frequency space."""
    f = abs(f_proj)
    #a = zinfmag*np.exp(1j*zinfarg)*np.exp(-2j*np.pi*fr*(tau+1j*Imtau))
    S21 = zinfmag*np.exp(1j*zinfarg)*np.exp(-2j*np.pi*(f-fr)*(tau+1j*Imtau))*(1-((Qr*(1/Qc+1j/Qm))/(1+(2j*Qr*(f-fr)/fr))))
    S21 = np.array(S21)
    real_S21 = S21.real
    real_S21[f_proj>0] = 0
    imag_S21 = S21.imag
    imag_S21[f_proj<0] = 0
    return real_S21 + imag_S21

def resfunc8(f, fr, Qr, Qc, a, Qm, tau):
    """A semi-obvious form of Gao's S21 function. e^(2j*pi*fr*tau) is incorporated into a."""
    S21 = a*np.exp(-2j*np.pi*(f-fr)*tau)*(1-((Qr*(1/Qc+1j/Qm))/(1+(2j*Qr*(f-fr)/fr))))
    return S21

def finefit(f, z, fr_0, tau_0, fwindow, numspan=2, resnum=None, plotting=True):
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
    fr_1, Qr_1, phi_1, zd, zinf, Qc_1, Qi_1, Qi0, Qm_0 = roughfit(f, z, fr_0, tau_0, numspan=numspan, resnum=resnum, plotting=False)

    # trim data?
    if False:
        fnew, znew = trimdata(f, z, fr_1, Qr_1, numspan=numspan)
        if len(fnew)>10:
            f = fnew
            z = znew

    # combine x and y data so the fit can go over both simultaneously
    xdata = np.concatenate([-f, f])
    ydata = np.concatenate([z.real, z.imag])

    a_1 = zinf*np.exp(-2j*np.pi*fr_1*(-tau_0))
    phi_2 = np.angle(-zd/zinf, deg=False)
    Qm_1 = 1./(np.exp(1.j*phi_2)/Qc_1).imag
    Qc_2 = 1./(np.exp(1.j*phi_2)/Qc_1).real
    Qc_2 = Qc_1
    #print Qm_1, Qc_2
    #phi_2 = (-1/zinf).imag

    def resfunc6(f_proj, Imtau):
        """An alternate resfunc3 with all real inputs. Re(S21) is projected onto negative frequency space."""
        f = abs(f_proj)
        #a = abs(zinf)*np.exp(1j*np.angle(zinf))*np.exp(-2j*np.pi*fr_1*(tau_0+1j*Imtau))
        S21 = abs(zinf)*np.exp(1j*np.angle(zinf))*np.exp(-2j*np.pi*(f-fr_1)*(tau_0+1j*Imtau))*(1-(((Qr_1/Qc_1)*np.exp(1j*phi_2))/(1+(2j*Qr_1*(f-fr_1)/fr_1))))
        #S21 = np.array(S21)
        real_S21 = S21.real
        real_S21[f_proj>0] = 0
        imag_S21 = S21.imag
        imag_S21[f_proj<0] = 0
        return real_S21 + imag_S21
    fparams, fcov = opt.curve_fit(resfunc6, xdata, ydata, p0=0, bounds=(-200, 200))
    Imtau_0 = fparams[0]

    def resfunc9(f_proj,Qc,Qm):
        f = abs(f_proj)
        #a = abs(zinf)*np.exp(1j*np.angle(zinf))*np.exp(-2j*np.pi*fr_1*(tau_0+1j*Imtau))
        S21 = abs(zinf)*np.exp(1j*np.angle(zinf))*np.exp(-2j*np.pi*(f-fr_1)*(tau_0+1j*Imtau_0))*(1-(Qr_1*(1/Qc+1j/Qm))/(1+(2j*Qr_1*(f-fr_1)/fr_1)))
        #S21 = np.array(S21)
        real_S21 = S21.real
        real_S21[f_proj>0] = 0
        imag_S21 = S21.imag
        imag_S21[f_proj<0] = 0
        return real_S21 + imag_S21
    #fparams, fcov = opt.curve_fit(resfunc9, xdata, ydata, p0=[Qc_1,Qm_1])
    #Qc_2,Qm_2 = fparams
    #plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(ydata[:len(z)]+1j*ydata[len(z):])),'.')
    #plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc9(xdata[:len(z)],Qc_1,Qm_1)+1j*resfunc9(xdata[len(z):],Qc_1,Qm_1))), label='1')
    #plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc9(xdata[:len(z)],Qc_2,Qm_2)+1j*resfunc9(xdata[len(z):],Qc_2,Qm_2))), label='2')
    #plt.plot(xdata,ydata,label='data')
    #plt.plot(xdata,resfunc9(xdata,Qc_1,Qm_1),label='1')
    #plt.plot(xdata,resfunc9(xdata,Qc_2,Qm_2),label='2')
    #plt.legend()
    #plt.show()

    #plotting=True
    if plotting:
        plt.figure(1)
        plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(ydata[:len(z)]+1j*ydata[len(z):])),'.')
        #plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc5(xdata[:len(z)],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, 0)+1j*resfunc5(xdata[len(z):],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, 0))), label='roughfit, Imtau=0')
        plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc5(xdata[:len(z)],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, Imtau_0)+1j*resfunc5(xdata[len(z):],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, Imtau_0))), label='roughfit with phi')
        plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc7(xdata[:len(z)],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), Qm_1, tau_0, Imtau_0)+1j*resfunc7(xdata[len(z):],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), Qm_1, tau_0, Imtau_0))), label='roughfit with Qm_1')
        plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc7(xdata[:len(z)],fr_1, Qr_1, Qc_2, abs(zinf), np.angle(zinf), Qm_2, tau_0, Imtau_0)+1j*resfunc7(xdata[len(z):],fr_1, Qr_1, Qc_2, abs(zinf), np.angle(zinf), Qm_2, tau_0, Imtau_0))), label='roughfit with Qm_2')
        #plt.plot(abs(xdata[:len(z)]),20*np.log10(np.abs(resfunc7(xdata[:len(z)],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), Qm_0, tau_0, Imtau_0)+1j*resfunc7(xdata[len(z):],fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), Qm_0, tau_0, Imtau_0))), label='roughfit')
        plt.legend()
        #plt.show()

    #print 'fr_1 = ' + str(fr_1)
    #print 'Qr_1 = ' + str(Qr_1)
    #print 'Qc_1 = ' + str(Qc_1)
    #print 'tau_0 = ' + str(tau_0+1j*Imtau_0)
    #print 'phi_2 = ' + str(phi_2) # This number would preferable be between -0.5*pi and +0.5*pi

    #fparams, fcov = opt.curve_fit(resfunc7, xdata, ydata, p0=[fr_1, Qr_1, Qc_2, abs(zinf), np.angle(zinf), Qm_2, tau_0, Imtau_0], bounds=([(fr_1-fwindow), 0, 0, 0, -2*np.pi,-np.inf, -200, -200], [(fr_1+fwindow), np.inf, np.inf, np.inf, 2*np.pi, np.inf, 200, 200]), sigma=abs(abs(xdata)-np.mean(f)+0.000000000001)**0.5)
    fparams, fcov = opt.curve_fit(resfunc5, xdata, ydata, p0=[fr_1, Qr_1, Qc_1, abs(zinf), np.angle(zinf), phi_2, tau_0, Imtau_0], bounds=([(fr_1-fwindow), 0, 0, 0, -2*np.pi,-2*np.pi, -200, -200], [(fr_1+fwindow), np.inf, np.inf, np.inf, 2*np.pi, 2*np.pi, 200, 200]))#, sigma=abs(abs(xdata)-np.mean(f)+0.000000000001)**0.5)
    #print fparams[5], fparams[2]
    #print fparams[1], fparams[2], fparams[1]*fparams[2]/(fparams[2]-fparams[1]), fparams[5]
    print ''

    if plotting:
        plt.plot(abs(xdata[:len(z)]), 20*np.log10(np.abs(resfunc7(xdata[:len(z)],fparams[0],fparams[1],fparams[2],fparams[3],fparams[4],fparams[5],fparams[6],fparams[7])+1j*resfunc7(xdata[len(z):],fparams[0],fparams[1],fparams[2],fparams[3],fparams[4],fparams[5],fparams[6],fparams[7]))), label='finefit')
        plt.plot(fparams[0], 20*np.log10(np.abs(resfunc7(-1.*fparams[0],fparams[0],fparams[1],fparams[2],fparams[3],fparams[4],fparams[5],fparams[6],fparams[7])+1j*resfunc7(fparams[0],fparams[0],fparams[1],fparams[2],fparams[3],fparams[4],fparams[5],fparams[6],fparams[7]))), label='finefit fr', marker='o')
        plt.legend()
        plt.figure(2)
        plt.plot(abs(xdata),ydata,'.')
        plt.plot(abs(xdata),resfunc7(xdata,fparams[0],fparams[1],fparams[2],fparams[3],fparams[4],fparams[5],fparams[6],fparams[7]))
        plt.show()

    fr_fine = fparams[0]
    Qr_fine = fparams[1]
    Qc_hat_mag_fine = fparams[2]
    a_fine = fparams[3]*np.exp(1j*fparams[4])#*np.exp(2j*np.pi*fparams[0]*(fparams[6]+1j*fparams[7]))
    phi_fine = fparams[5]
    tau_fine = fparams[6] + 1j*fparams[7]

    # find Qc, unsure where this equation originates
    #Qc = (abs(zinf)/abs(zd))*Q
    # find Qi (all Q that isn't Qc)
    Qi_fine = 1/((1/Qr_fine)-(1/ Qc_hat_mag_fine))
    #print Qi_fine

    # the next section finds Qi0 and other parameters. I'm not sure where any of these equations come from or why we want these parameters
    zf0 = zinf + zd
    # projection of zf0 into zinf
    l = (zf0*np.conj(zinf)).real/abs(zinf)**2
    # Q/Qi0 = l
    Qi0 = Qr_1/l
    zc = zinf + zd/2
    r = abs(zd/2)
    phi = np.angle(-zd/zc, deg=False)

    Qc_fine =  Qc_hat_mag_fine/np.cos(phi_fine)

    #print  Qc_fine, 1/np.real(np.exp(1j*Qm_fine)/Qc_fine)

    return fr_fine, Qr_fine, Qc_hat_mag_fine, a_fine, phi_fine, tau_fine, Qc_fine


def sweep_fit_from_file(fname, nsig=3, fwindow=5e-4, chan="S21", h5_rewrite=False, pdf_rewrite=False, additions=[], start_f=None, stop_f=None):
    """
    inputs save_scatter data to sweep_fit, can save the results back to the h5 file

    Input parameters:
              fname: full name of the save_scatter h5 file
               nsig: nsig*sigma is the threshold for resonator identification
            fwindow: half the window cut around each resonator before fitting [GHz]
               chan: channel from save_scatter being analyzed
         h5_rewrite: save fit data to the filename.h5?
        pdf_rewrite: save fit data to the filename.pdf?
          additions: list of resonances to be manually included [GHz]
            start_f: lower bound of resonance identification region [GHz]
             stop_f: upper bound of resonance identification region [GHz]

    Returns:
        fr_list, Qr_list, Qc_list, Qi_list (fitted values for each resonator)
    """

    with h5py.File(fname, "r") as fyle:
        f = np.array(fyle["{}/f".format(chan)])
        z = np.array(fyle["{}/z".format(chan)])
    fr_list, Qr_list, Qc_list, Qi_list = sweep_fit(f,z,nsig=nsig,fwindow=fwindow,pdf_rewrite=pdf_rewrite,additions=additions,filename=fname[:-3],start_f=start_f,stop_f=stop_f)

    # save the lists to fname
    if h5_rewrite == True:
        with h5py.File(fname, "r+") as fyle:
            if "{}/fr_list".format(chan) in fyle:
                fyle.__delitem__("{}/fr_list".format(chan))
            if "{}/Qr_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qr_list".format(chan))
            if "{}/Qc_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qc_list".format(chan))
            if "{}/Qi_list".format(chan) in fyle:
                fyle.__delitem__("{}/Qi_list".format(chan))
            fyle["{}/fr_list".format(chan)] = fr_list
            fyle["{}/Qr_list".format(chan)] = Qr_list
            fyle["{}/Qc_list".format(chan)] = Qc_list
            fyle["{}/Qi_list".format(chan)] = Qi_list

def sweep_fit(f, z, nsig=3, fwindow=5e-4, pdf_rewrite=False, additions=[], filename='test', start_f=None, stop_f=None):
    """
    sweep_fit fits data to the resonator model described in Jiansong's thesis

    Input parameters:
                  f: array of frequency values [GHz]
                  z: array of corresponding z values
               nsig: nsig*sigma is the threshold for resonator identification
            fwindow: half the window cut around each resonator before fitting [GHz]
        pdf_rewrite: save fit data to the filename.pdf?
          additions: list of resonances to be manually included [GHz]
           filename: name used in the output pdf file
            start_f: lower bound of resonance identification region [GHz]
             stop_f: upper bound of resonance identification region [GHz]

    Returns:
        fr_list, Qr_list, Qc_list, Qi_list (fitted values for each resonator)
    """

    nfreq = 1/(2*(abs(f[-1]-f[0])/(len(f)-1)))    # The nyquist frequency [s]
    evfreq = 1/(2*fwindow)    # The frequency corresponding to the expected window size [s]
    b, a = sig.butter(2, evfreq/nfreq, btype='highpass')
    mfz = np.sqrt(sig.filtfilt(b, a, z.real)**2 + sig.filtfilt(b, a, z.imag)**2)  # The magnitude of filtered z, The filtfilt part calls a deprication warning for unknown reasons

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

    # initialize mx below min
    mx = -np.inf
    peak_pos = 0
    mx_pos = np.nan
    lookformax = False
    delta = nsig*bstd
    gamma = 3*np.mean(mfz[mfz<delta])

    # find peaks and add them to peaklist
    for i in range(len(mfz)):
        if (f[i] >= start_f)*(f[i] <= stop_f):
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
                if cp > delta and f[i] > (min(f)+2*fwindow):
                    mx = cp
                    mx_pos = i
                    lookformax = True

    peaklist = sorted(peaklist)
    print 'peaklist', peaklist

    # Plot the transmission and peaks
    plt.figure()

    fig, axarr = plt.subplots(nrows=2, sharex=True, num=1)
    axarr[0].plot(f, 20*np.log10(np.abs(np.array(z)))) # plot the unaltered transmission
    axarr[0].set_title('Transmission with Resonance Identification')
    axarr[0].set_ylabel("|$S_{21}$| [dB]")
    axarr[1].plot(f, mfz/bstd)
    axarr[1].set_ylabel("|filtered z| [#std]")
    axarr[1].set_xlabel("Frequency [GHz]")
    axarr[1].axhline(y=nsig, color="red", label="nsig = "+str(nsig))
    axarr[1].axhline(y=gamma/bstd, color="green")
    axarr[1].axvline(x=start_f, color="gray")
    axarr[1].axvline(x=stop_f, color="gray")
    axarr[1].plot(f[peaklist], mfz[peaklist]/bstd, 'gs', label=str(len(peaklist)-len(additions))+" resonances identified")
    axarr[1].plot(f[addlist], mfz[addlist]/bstd, 'ys', label=str(len(addlist))+" resonances manually added")
    plt.legend()

    plt.show()

    # Save to pdf if pdf_rewrite == True
    if pdf_rewrite == True:
        plt.figure(figsize=(10, 10))

        fig, axarr = plt.subplots(nrows=2, sharex=True, num=1)
        axarr[0].plot(f, 20*np.log10(np.abs(np.array(z)))) # plot the unaltered transmission
        axarr[0].set_title('Transmission with Resonance Identification')
        axarr[0].set_ylabel("|$S_{21}$| [dB]")
        axarr[1].plot(f, mfz/bstd)
        axarr[1].set_ylabel("|filtered z| [#std]")
        axarr[1].set_xlabel("Frequency [GHz]")
        axarr[1].axhline(y=nsig, color="red", label="nsig = "+str(nsig))
        axarr[1].axhline(y=gamma/bstd, color="green")
        axarr[1].axvline(x=start_f, color="gray")
        axarr[1].axvline(x=stop_f, color="gray")
        axarr[1].plot(f[peaklist], mfz[peaklist]/bstd, 'gs', label=str(len(peaklist)-len(additions))+" resonances identified")
        axarr[1].plot(f[addlist], mfz[addlist]/bstd, 'ys', label=str(len(addlist))+" resonances manually added")
        plt.legend()

        Res_pdf = PdfPages(filename+'.pdf')
        Res_pdf.savefig()
        plt.close()

    # initialize the parameter lists
    fr_list = np.zeros(len(peaklist))
    Qr_list = np.zeros(len(peaklist))
    Qc_hat_mag_list = np.zeros(len(peaklist))
    Qc_list = np.zeros(len(peaklist))
    Qi_list = np.zeros(len(peaklist))
    a_list = np.array([0.+0j]*len(peaklist))
    phi_list = np.zeros(len(peaklist))
    tau_list = np.array([0.+0j]*len(peaklist))

    # define the windows around each peak. and then use finefit to find the parameters
    for i in range(len(peaklist)):
        print 'Resonance #{}'.format(str(i))
        curr_pts = (f >= (f[peaklist[i]]-fwindow)) & (f <= (f[peaklist[i]]+fwindow))
        f_curr = f[curr_pts]
        z_curr = z[curr_pts]
        #print np.vstack((f_curr,z_curr)).T
        z_curr_1 = [zs for _,zs in sorted(zip(f_curr,z_curr))]
        f_curr_1 = [fs for fs,_ in sorted(zip(f_curr,z_curr))]
        f_curr = np.array(f_curr_1)
        z_curr = np.array(z_curr_1)

        try:
            fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i], Qc_list[i] = finefit(f_curr, z_curr, f[peaklist[i]], 30, fwindow, numspan=2, resnum=i, plotting=False)
            Qi_list[i] = (Qr_list[i]*Qc_list[i])/(Qc_list[i]-Qr_list[i])
        except:
            print '      failure'
            fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i], Qc_list[i] = [0,0,0,0,0,0,0]
            Qi_list[i] = 0

        fit_at_input = resfunc3(f_curr, fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
        fitres_Chi2 = sum((z_curr.real-fit_at_input.real)**2+(z_curr.imag-fit_at_input.imag)**2)

        if pdf_rewrite == True:
            f_continuous = np.linspace(f_curr[0],f_curr[-1],14*len(f_curr))
            fit = resfunc3(f_continuous, fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            zrfit = resfunc3(fr_list[i], fr_list[i], Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            fit_down = resfunc3(f_continuous, fr_list[i], 0.95*Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            fit_up = resfunc3(f_continuous, fr_list[i], 1.05*Qr_list[i], Qc_hat_mag_list[i], a_list[i], phi_list[i], tau_list[i])
            fitwords = "$f_{r}$ = " + str(fr_list[i]) + "\n" + "$Q_{r}$ = " + str(Qr_list[i]) + "\n" + "$Q_{c}$ = " + str(Qc_list[i]) + "\n" + "$Q_{i}$ = " + str((Qr_list[i]*Qc_list[i])/(Qc_list[i]-Qr_list[i])) + "\n" + "$\phi_{0}$ = " + str(phi_list[i]) + "\n" + "$a$ = " + str(a_list[i]) + "\n" + r"$\tau$ = " + str(tau_list[i]) + "\n"

            plt.figure(figsize=(10, 10))

            plt.subplot(2,2,1)
            plt.plot(f_curr, 20*np.log10(np.abs(z_curr)),'.', label='Data')
            #plt.plot(f_continuous, 20*np.log10(np.abs(fit_down)), label='Fit 0.95Q')
            plt.plot(f_continuous, 20*np.log10(np.abs(fit)), label='Fit 1.00Q', color='red')
            #plt.plot(f_continuous, 20*np.log10(np.abs(fit_up)), label='Fit 1.05Q')
            plt.plot(fr_list[i], 20*np.log10(np.abs(zrfit)), '*', markersize=10, color='red', label='$f_{r}$')
            plt.title("resonance " + str(i) + " at " + str(int(10000*fr_list[i])/10000) + " GHz")
            plt.xlabel("Frequency [GHz]")
            plt.xticks([min(f_curr),max(f_curr)])
            plt.ylabel("|$S_{21}$| [dB]")
            plt.legend(bbox_to_anchor=(2, -0.15))

            plt.subplot(2,2,2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axvline(x=0, color='gray')
            plt.axhline(y=0, color='gray')
            plt.plot(z_curr.real, z_curr.imag,'.', label='Data')
            #plt.plot(fit_down.real, fit_down.imag,  label='Fit 0.95Q')
            plt.plot(fit.real, fit.imag,  label='Fit 1.00Q', color='red')
            #plt.plot(fit_up.real, fit_up.imag, label='Fit 1.05Q')
            plt.plot(zrfit.real, zrfit.imag, '*', markersize=10, color='red',  label='Fr')
            plt.xlabel("$S_{21}$ real")
            plt.xticks([min(z_curr.real),max(z_curr.real)])
            plt.ylabel("$S_{21}$ imaginary",labelpad=-40)
            plt.yticks([min(z_curr.imag),max(z_curr.imag)])

            plt.figtext(0.55, 0.085, fitwords)
            #plt.figtext(0.75, 0.14, fitwords2)
            plt.figtext(0.5, 0.26, r"$S_{21}(f)=ae^{-2\pi j(f-fr)\tau}\left [ 1-\frac{\frac{Q_{r}}{|\widehat{Q}_{c}|}e^{j\phi_{0}}}{1+2jQ_{r}(\frac{f-f_{r}}{f_{r}})} \right ]$", fontsize=20)

            plt.subplot(2,2,3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axvline(x=0, color='gray')
            plt.axhline(y=0, color='gray')
            zi_no_cable = removecable(f_curr, z_curr, tau_list[i], fr_list[i])/(a_list[i])
            zi_normalized = 1-((1 - zi_no_cable)*np.cos(phi_list[i])/np.exp(1j*(phi_list[i])))
            plt.plot(zi_normalized.real, zi_normalized.imag,'.')
            zfit_no_cable = removecable(f_continuous, fit, tau_list[i], fr_list[i])/(a_list[i])
            zfit_normalized = 1-((1 - zfit_no_cable)*np.cos(phi_list[i])/np.exp(1j*(phi_list[i])))
            plt.plot(zfit_normalized.real, zfit_normalized.imag, color='red')
            zrfit_no_cable = removecable(fr_list[i], zrfit, tau_list[i], fr_list[i])/(a_list[i])
            zrfit_normalized = 1-((1 - zrfit_no_cable)*np.cos(phi_list[i])/np.exp(1j*(phi_list[i])))
            plt.plot(zrfit_normalized.real, zrfit_normalized.imag,'*', markersize=10, color='red')

            Res_pdf.savefig()
            plt.close()

    if pdf_rewrite == True:
        Res_pdf.close()

    return fr_list, Qr_list, Qc_list, Qi_list

if __name__ == '__main__':
    sweep_fit_from_file("191015OW190920p1.h5", nsig=2, fwindow=5e-4, h5_rewrite=False, pdf_rewrite=False, start_f=3, stop_f=3.6)
    #sweep_fit_from_file("191018OW190920p1.h5", nsig=2.1, fwindow=5e-4, h5_rewrite=False, pdf_rewrite=False, start_f=3, stop_f=3.6)
    plt.show()
