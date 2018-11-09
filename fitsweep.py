import fitres
import h5py as h5
import pandas as pd





def fit_all_sweeps(fname="181019YY180726p2.h5", sweeptype="tempsweep"):
    with h5.File(fname) as fyle:
        result = pd.DataFrame()
        for sdate in fyle[sweeptype].keys():
            for sweep in fyle[sweeptype][sdate].keys():
                result =  result.append(fit_single_sweep(fname, sweeptype, sdate, sweep))
    return result    


def fit_single_sweep(fname, sweeptype, sdate, sweep):
    with h5.File(fname) as fyle:
        fr_list_0 = fyle["S21/fr_list"][:]

        df = pd.read_hdf(fname, "{}/{}/{}".format(sweeptype, sdate, sweep))
        fr_list, Qr_list, Qc_list, a_list, phi_list, tau_list = [], [], [], [], [], []
        for resID, fr_nominal in enumerate(fr_list_0):
            df_temp = df[df.resID == resID]
            fwindow = 5e-4 # (max(df_temp.f) - min(df_temp.f)) / 2.0
            f_curr = df_temp.f.values
            z_curr = df_temp.z.values
            print f_curr
            print z_curr

            fr_list[resID], Qr_list[resID], Qc_list[resID], a_list[resID], phi_list[resID], tau_list[resID] = fitres.finefit(f_curr, z_curr, fr_nominal, 30, fwindow, numspan=2)

        fyle[sweeptype][sdate]
    result = pd.DataFrame({"fr_list":fr_list, "Qr_list":Qr_list, "Qc_list":Qc_list, "a_list":a_list, "phi_list":phi_list, "tau_list":tau_list})
    result["sweep"] = float(sweep)
    return result