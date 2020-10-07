from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def resfunc3(f, fr, Qr, Qc_hat_mag, a, phi, tau):
    """A semi-obvious form of Gao's S21 function. e^(2j*pi*fr*tau) is incorporated into a."""
    S21 = a*np.exp(-2j*np.pi*(f-fr)*tau)*(1-(((Qr/Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return S21

fig, ax = plt.subplots(ncols=2)
ax[1].axvline(x=0,c='gray')
ax[1].axhline(y=0,c='gray')
ax[1].set_aspect('equal', 'box')
plt.subplots_adjust(left=0.25, bottom=0.3)

freqs = np.linspace(3.0, 4.0, 1000)
Qr_0 = 100
phi_0 = 0
Qc_hat_mag_0 = 150

S21_0 = resfunc3(freqs,3.5,Qr_0,Qc_hat_mag_0,1,phi_0,0)
dots_0 = resfunc3(np.array([3.5*(1-1/Qr_0),3.5,3.5*(1+1/Qr_0)]),3.5,Qr_0,Qc_hat_mag_0,1,phi_0,0)
abs_plot, = ax[0].plot(freqs, 20*np.log10(abs(S21_0)), lw=2, c='blue')
com_plot, = ax[1].plot(S21_0.real, S21_0.imag,'.', c='blue')
abs_dots, = ax[0].plot(np.array([3.5*(1-1/Qr_0),3.5,3.5*(1+1/Qr_0)]),20*np.log10(abs(dots_0)),'o')
com_dots, = ax[1].plot(dots_0.real,dots_0.imag,'o')
ax[0].margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_Qr = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_Qc_hat_mag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_phi = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

s_Qr = Slider(ax_Qr, '$Qr$', 10, 1000, valinit=Qr_0)
s_Qc_hat_mag = Slider(ax_Qc_hat_mag, '$|\widehat{Q}_{c}|$', 10, 1000, valinit=Qc_hat_mag_0)
s_phi = Slider(ax_phi, '$\phi$', -1*np.pi, np.pi, valinit=phi_0)


def update(val):
    Qr_1 = s_Qr.val
    phi_1 = s_phi.val
    Qc_hat_mag_1 = s_Qc_hat_mag.val
    dots_1 = resfunc3(np.array([3.5*(1-1/Qr_1),3.5,3.5*(1+1/Qr_1)]),3.5,Qr_1,Qc_hat_mag_1,1,phi_1,0)
    if np.cos(phi_1)<=0: # negative Qc
        abs_plot.set_color('red')
        com_plot.set_color('red')
    elif Qr_1 > s_Qc_hat_mag.val/np.cos(phi_1): # negative Qi
        abs_plot.set_color('orange')
        com_plot.set_color('orange')
    else:
        abs_plot.set_color('blue')
        com_plot.set_color('blue')
    S21_1 = resfunc3(freqs,3.5,Qr_1,Qc_hat_mag_1,1,phi_1,0)
    abs_plot.set_ydata(20*np.log10(abs(S21_1)))
    com_plot.set_xdata(S21_1.real)
    com_plot.set_ydata(S21_1.imag)
    abs_dots.set_xdata(np.array([3.5*(1-1/Qr_1),3.5,3.5*(1+1/Qr_1)]))
    abs_dots.set_ydata(20*np.log10(abs(dots_1)))
    com_dots.set_xdata(dots_1.real)
    com_dots.set_ydata(dots_1.imag)
    fig.canvas.draw_idle()


s_Qr.on_changed(update)
s_Qc_hat_mag.on_changed(update)
s_phi.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_Qr.reset()
    s_Qc_hat_mag.reset()
    s_phi.reset()
button.on_clicked(reset)

#rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    abs_plot.set_color(label)
    com_plot.set_color(label)
    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)

plt.show()
