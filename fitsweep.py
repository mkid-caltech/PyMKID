import fitres
import h5py as h5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math


def fit_all_sweeps(fname="181019YY180726p2.h5", sweeptype="powsweep", h5_rewrite = False):
    with h5.File(fname) as fyle:
        result = pd.DataFrame()
        for sdate in fyle[sweeptype].keys():
            for sweep in fyle[sweeptype][sdate].keys():
                result =  result.append(fit_single_sweep(fname, sweeptype, sdate, sweep, h5_rewrite))
    return result


def fit_single_sweep(fname, sweeptype, sdate, sweep, h5_rewrite = False):
    with h5.File(fname) as fyle:
        fr_list_0 = fyle["S21/fr_list"][:]

        df = pd.read_hdf(fname, "{}/{}/{}".format(sweeptype, sdate, sweep))

        num_res = len(fr_list_0)
        fr_list, Qr_list, Qc_mag_list, a_list, phi_list, tau_list, Qc_list = \
        [None]*num_res, [None]*num_res, [None]*num_res, [None]*num_res, [None]*num_res, [None]*num_res, [None]*num_res
        for resID, fr_nominal in enumerate(fr_list_0):
            print sweep, fr_nominal
            df_temp = df[df.resID == resID]
            fwindow = 5e-4 # (max(df_temp.f) - min(df_temp.f)) / 2.0
            f_curr = df_temp.f.values
            z_curr = df_temp.z.values
            # print f_curr
            # print z_curr
            try:
                fr_list[resID], \
                Qr_list[resID], \
                Qc_mag_list[resID], \
                a_list[resID], \
                phi_list[resID], \
                tau_list[resID], \
                Qc_list[resID] = \
                fitres.finefit(f_curr, z_curr, fr_nominal)
            except:
                print "couldn't find a fit"
                fr_list[resID], Qr_list[resID], Qc_mag_list[resID], a_list[resID], phi_list[resID], tau_list[resID], Qc_list[resID] = \
                0, 0, 0, 0, 0, 0, 0
        if h5_rewrite == True:
            with h5.File(fname, "r+") as fyle:
                if "{}/{}/{}/fr_list".format(sweeptype,sdate,sweep) in fyle:
                    fyle.__delitem__("{}/{}/{}/fr_list".format(sweeptype,sdate,sweep))
                if "{}/{}/{}/Qr_list".format(sweeptype,sdate,sweep) in fyle:
                    fyle.__delitem__("{}/{}/{}/Qr_list".format(sweeptype,sdate,sweep))
                if "{}/{}/{}/Qc_list".format(sweeptype,sdate,sweep) in fyle:
                    fyle.__delitem__("{}/{}/{}/Qc_list".format(sweeptype,sdate,sweep))


                fyle["{}/{}/{}/fr_list".format(sweeptype,sdate,sweep)] = fr_list
                fyle["{}/{}/{}/Qr_list".format(sweeptype,sdate,sweep)] = Qr_list
                fyle["{}/{}/{}/Qc_list".format(sweeptype,sdate,sweep)] = Qc_list
    result = pd.DataFrame({"fr_list":fr_list, "Qr_list":Qr_list, "Qc_mag_list":Qc_mag_list, "a_list":a_list, "phi_list":phi_list, "tau_list":tau_list, "Qc_list": Qc_list})
    result["sweep"] = float(sweep)
    return result

def fit_temp_pow_sweep(fname):
    """
    Osmond Wen, December 10, 2019
    decription: finds the resonant frequencies from the combined temperature and
                power sweep performed by powsweep.sweep_temp_pow
    """
    fr, Qr = {}, {}
    with h5.File(fname) as fyle:
        fr_list_0 = fyle["S21/fr_list"][:]
        full_location = "tempsweep/2019-12-06-14-35-28"
        for temperature in np.array(fyle[full_location].keys()):
            print 'Temperature: ', temperature
            powernams = np.array(fyle[full_location+"/"+temperature].keys())
            powervals = powernams.astype(np.float)
            powernams = powernams[powervals.argsort()]
            fr[float(temperature)] = {}
            Qr[float(temperature)] = {}
            for power in powernams:
                print '    Power: ', power
                df = pd.read_hdf(fname, key=full_location+"/"+temperature+"/"+power)
                fr[float(temperature)][float(power)] = {}
                Qr[float(temperature)][float(power)] = {}
                for resID, fr_nominal in enumerate(fr_list_0):
                    print '         Resonator: ', resID
                    df_thisKID = df[df.resID == resID]
                    f_thisKID = df_thisKID.f.values
                    z_thisKID = df_thisKID.z.values
                    fr[float(temperature)][float(power)][resID],\
                    Qr[float(temperature)][float(power)][resID],\
                    _,_,_,_,_ = \
                    fitres.finefit(f_thisKID, z_thisKID, fr_nominal)
    return fr, Qr

def plot_Osmond(fr,Qr):
    """
    Osmond Wen, December 10, 2019
    description: used to plot things.
                 2019.12.10: plotting the shift in the resonant frequency as a
                             function of temperature
                 2019.12.10: plotting the shift in the quality factor as a
                             function of temperature
    input: fr, which is a nested dictionary that comes from fit_temp_pow_sweep
    """
    should_plot_fr = True
    delta_f_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    delta_Q_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for temperature in fr:
        for power in fr[temperature]:
            for resID in fr[temperature][power]:
                delta_f_curves[resID][power][temperature] = \
                (fr[temperature][power][resID] - fr[.06][-28][resID])

                # delta_Q_curves[resID][power][temperature] = \
                # (Qr[temperature][power][resID] - Qr[.06][-28][resID]) / \
                # Qr[.06][-28][resID]

                delta_Q_curves[resID][power][temperature] = \
                (1/Qr[temperature][power][resID] - 1/Qr[.06][-28][resID])

    powers =  delta_f_curves[0].keys()
    powers.sort()
    N = len(delta_f_curves.keys())
    dim = int(math.ceil(math.sqrt(resID)))

    plt.close('all')
    f_fig, f_axes = plt.subplots(dim,dim)
    Q_fig, Q_axes = plt.subplots(dim,dim)
    for resID in delta_f_curves:
        for power in powers:
            f_data = delta_f_curves[resID][power].items()
            temps  = [1E3*x[0] for x in f_data]
            del_f = [1E6*x[1] for x in f_data]

            Q_data = delta_Q_curves[resID][power].items()
            del_Q = [x[1] for x in Q_data]

            p_row = int(math.floor(resID/dim))
            p_col = int(resID % dim)

            f_axes[p_row,p_col].plot(temps, del_f,'.')
            f_axes[p_row,p_col].set_title('resonator: ' + str(int(resID + 1)))

            Q_axes[p_row,p_col].plot(temps, del_Q,'.')
            Q_axes[p_row,p_col].set_title('resonator: ' + str(int(resID + 1)))
        f_fig.legend(powers)
        f_fig.text(0.5, 0.04,'temperature (mK)', ha='center')
        f_fig.text(0.04, 0.5,'change in resonant frequency (kHz)', va='center', rotation='vertical')

        Q_fig.legend(powers)
        Q_fig.text(0.5, 0.04,'temperature (mK)', ha='center')
        Q_fig.text(0.04, 0.5,'change in 1/Q', va='center', rotation='vertical')
    plt.show(False)


    should_plot_Qr = False
