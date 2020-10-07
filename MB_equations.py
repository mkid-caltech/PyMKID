import numpy as np
import iminuit
import scipy.special as spec

#############################
# iMinuit fitting functions #
#############################

######################
# Mattis-Bardeen fit #
######################

Boltz_k = 8.6173303E-5 # eV/K
N_0 = 1.72E28 # 1/(m^3*eV), Single-spin density of states (aluminum, from Jiansong's Thesis)
Planck_h = 4.135667662E-15 # eV*s

def signed_log10(x):
	return np.log10(np.abs(x)) * x/np.abs(x)

def n_qp(T, Delta0):
	# [K, eV]
	return 2.*N_0*np.sqrt(2.*np.pi*Boltz_k*T*Delta0)*np.exp(-1.*Delta0/(Boltz_k*T))

def f_T(T, f0, Delta0, alpha_f):
	# [K, Hz, eV, _]
	xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
	return -1.*alpha_f/(4.*Delta0*N_0) * ( 1. + np.sqrt((2.*Delta0)/(np.pi*Boltz_k*T)) * np.exp(-1.*xi) * spec.i0(xi) ) * n_qp(T,Delta0) * f0 + f0

def Qi_T(T, f0, Qi0, Delta0, alpha_Q):
	xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
	return ( alpha_Q/(np.pi*N_0) * np.sqrt(2./(np.pi*Boltz_k*T*Delta0)) * np.sinh(xi) * spec.k0(xi) * n_qp(T,Delta0) + 1./Qi0 )**-1.

def kappa_1(T, f0, Delta0):
	xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
	return (1/(np.pi*N_0))*np.sqrt(2./(np.pi*Boltz_k*T*Delta0))*np.sinh(xi)*spec.k0(xi)

def kappa_2(T, f0, Delta0):
	xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
	return (1/(2.*Delta0*N_0))*(1.+np.sqrt((2.*Delta0)/(np.pi*Boltz_k*T))*np.exp(-1.*xi)*spec.i0(xi))

def S_1(fr,T,Delta):
	# [Hz, K, eV]
	xi = 1./2.*(Planck_h*fr)/(Boltz_k*T)
	return (2/np.pi)*np.sqrt(2*Delta/(np.pi*Boltz_k*T))*np.sinh(xi)*spec.k0(xi) # unitless

def S_2(fr,T,Delta):
	# [Hz, K, eV]
	xi = 1./2.*(Planck_h*fr)/(Boltz_k*T)
	return 1+np.sqrt(2*Delta/(np.pi*Boltz_k*T))*np.exp(-1*xi)*spec.i0(xi) # unitless

def MB_fitter(T_fit, Qi_fit, f_fit):

	fit_result = []

	def chisq(f0, Delta0, alpha, Qi0):
		alpha_Q = alpha
		alpha_f = alpha

		var_Qi = np.var(Qi_fit)
		var_f = np.var(f_fit)

		#return sum( (Qi_T(T_fit, f0, Qi0, Delta0, alpha_Q) - Qi_fit)**2./var_Qi + (f_T(T_fit, f0, Delta0, alpha_f) - f_fit)**2./var_f )
		return sum((f_T(T_fit, f0, Delta0, alpha_f) - f_fit)**2./var_f )

	def fit_chisq_test(T_fit, f_fit, Qi_fit, f0, Delta0, alpha, Qi0):
		var_Qi = np.var(Qi_fit)
		var_f = np.var(f_fit)

		return sum( (Qi_T(T_fit, f0, Qi0, Delta0, alpha) - Qi_fit)**2./var_Qi + (f_T(T_fit, f0, Delta0, alpha) - f_fit)**2./var_f )/4.

	f0_in = f_fit[0]
	Delta0_in = 4.e-4
	alpha_in = 0.05
	Qi0_in = Qi_fit[0]

	for j in range(100):
		minimizer = iminuit.Minuit(chisq, f0=f0_in, Delta0=Delta0_in, alpha=alpha_in, Qi0=Qi0_in, limit_f0=(f_fit[0]/1.1,f_fit[0]*1.1), limit_Delta0=(1.e-4,1.e-3), limit_alpha=(0.,0.5), limit_Qi0=(1.e2,1.e7), pedantic=False, print_level=-1)

		f0_in = minimizer.values["f0"]
		Delta0_in = minimizer.values["Delta0"]
		alpha_in = minimizer.values["alpha"]
		Qi0_in =minimizer.values["Qi0"]

		minimizer.migrad()

	f0 = minimizer.values["f0"]
	Delta0 = minimizer.values["Delta0"]
	alpha = minimizer.values["alpha"]
	Qi0 =minimizer.values["Qi0"]
	chi_sq_dof = fit_chisq_test(T_fit, f_fit, Qi_fit, f0, Delta0, alpha, Qi0)

	fit_result.append([f0/1.e9,Delta0*1000,alpha,Qi0,chi_sq_dof])

	T_smooth = np.linspace(T_fit[0],T_fit[-1],10000)

	return f0/1.e9, Delta0*1000., alpha, Qi0, chi_sq_dof
