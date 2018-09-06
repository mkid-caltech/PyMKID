import numpy as np
pi = np.pi
j = 1.j
k_B = 8.6173303E-5 #eV/K
h = 4.135667662e-15 #eV*s
e = 1.6021766208e-19 #C electron charge

# 30 nm Al film
N0 = 17.2 #nm^-3*eV^-1, J. Gao's thesis Eqn (2.87)~(2.90)
Delta = 0.18E-3 #eV, J. Gao's thesis Eqn (2.87)~(2.90)
R = 9.64E9 #nm^3/s, Sunil's email?


# YY180726.2
d = 30. #nm
T = 0.245 #K
f_r = 3.1142E9 #Hz, 1st resonance
V_sc = 4.176E13 #nm^3, 1st KID's inductor, not an accurate measurement
P_read = ((10**-3.5)/1000)/e #eV/s, converted from -35 dBm
alpha = 0.031 #Fabien's fit
Q_i = 1E4 #Fabien's fit
Q_qp = Q_i

n_qp_thermal = 2.*N0*np.sqrt(2.*pi*k_B*T*Delta)*np.exp(-Delta/(k_B*T)) #e.g. Sunil's note Eqn (240)
print "n_qp_thermal = %.3E nm^-3"%(n_qp_thermal)

from scipy.special import kv as Kv #modified Bessel function of the 2nd kind of real order v, Kv(v,x)
from scipy.special import iv as Iv #modified Bessel function of the 1st kind of real order v, Iv(v,x)

xi = h*f_r/(2*k_B*T)
kappa = (1./(2*N0*Delta))*((4./(pi*np.sqrt(2*pi*(k_B*T/Delta))))*np.sinh(xi)*Kv(0,xi)-j*(1.+np.sqrt(2*Delta/(pi*k_B*T))*np.exp(-xi)*Iv(0,xi))) #Sunil's note Eqn (85) (86)
kappa1 = kappa.real #Sunil's note Eqn (85) (86)
kappa2 = kappa.imag # nm^3
gamma1 = -1. #thin film limit, Sunil's note Eqn (91) (92)
gamma2 = 1.
print "\tgamma1 = %.3f\n\tgamma2 = %.3f\n\tkappa1 = %.3f nm^3\n\tkappa2 = %.3f nm^3"%(gamma1,gamma2,kappa1,kappa2)

n_qp_mean_Q = 1./(Q_qp*alpha*gamma1*kappa1) #Sunil's note Eqn (121)
n_qp_mean_fr = n_qp_mean_Q #use Q_qp-derived n_qp_mean to calculate df_r
df_r = (-1./2)*alpha*gamma2*kappa2*n_qp_mean_fr*f_r #Sunil's note Eqn (122)
## from Sunil's note Eqn (121) (122)
## 1/Q_qp = alpha*gamma1*kappa1*n_qp_mean
## df_r/f_r = -1./2*alpha*kappa2*n_qp_mean
print "n_qp_mean = %.3E nm^-3"%(n_qp_mean_Q)
print "df_r/f_r = %.3E, i.e. df_r = %.3f MHz"%(df_r/f_r, df_r/1E6)

n_qp_read = n_qp_mean_Q-n_qp_thermal
tau_max = 1/(R*n_qp_mean_Q)
eta_read = (n_qp_read*Delta*V_sc)/(tau_max*P_read)
## from Jonas' note Eqn (22)
## n_qp* = 1/R/tau_max
## n_qp_read = eta_read*P_read/Delta*tau_qp/V_sc
print "n_qp_read = %.3E nm^-3"%(n_qp_read)
print "eta_read = %.3E"%(eta_read)
