import glob
import numpy as np
import os
import scipy.signal
import json
import base64

def get_analytical(vsall):
    fs = 1000
    vs = np.copy(vsall)
    m = (vs[:,-1000:].ptp(1)<1) | np.isnan(vs).any(1)
    vs = vs[~m]
    vs = (vs - vs.mean(1)[:,None]) / vs.std(1)[:,None]
    vs[vs > 2] = 2
    vs[vs < -2] = -2
    sos = scipy.signal.butter(5, (2, 20), 'bp', fs=fs, output='sos')
    filt = scipy.signal.sosfiltfilt(sos, vs)
    analytic = scipy.signal.hilbert(filt - filt.mean())
    return np.angle(analytic)

def get_entropy(vsall, spike=None):
    tmp_phase = get_analytical(vsall)
    tmp_entropies = []
    #tmp_time = []

    #par_time_of_spike = list(recipe.spikes.keys())[0]

    if spike is None:
        r = range(0, tmp_phase.shape[1])
    else:
        r = range(5000-500, 5000+2000)

    for i in r: # just the spike
        if 1: # phase differences
            delta = tmp_phase[:,i,None] - tmp_phase[None,:,i]
            delta = (delta[np.triu_indices(len(delta), k = 1)] + np.pi) % (2*np.pi) - np.pi
        else: # just the phase
            delta = (tmp_phase[:,i] + np.pi) % (2*np.pi) - np.pi
        p, edges = np.histogram(delta, range=(-np.pi, np.pi), bins=30, density=True)
        x  = (edges[:-1]+edges[1:])/2
        S = -np.trapz(p * np.ma.log(p).filled(0), x)
        tmp_entropies.append(S)
        #tmp_time.append(t[i]/1e3)

    return np.array(tmp_entropies)

def fit_curve(t, S, time_peak):
    t = t[time_peak-500:time_peak+2500] / 1e3
    S = S[time_peak-500:time_peak+2500]
    peak = np.argmin(S)
    tmp_x = np.array(t[peak:])
    tmp_y = np.array(S[peak:])
    tmp_zero = tmp_y[-500:].mean()
    tmp_y = tmp_zero - tmp_y
    tmp_amp = tmp_zero - S[peak]
    (b,), _c = scipy.optimize.curve_fit(lambda x,b: tmp_amp*np.exp(-b*(x-tmp_x[0])),  tmp_x, tmp_y)
    tmp_Y = tmp_amp*np.exp(-b*(tmp_x-tmp_x[0]))
    decay_ms = 1000 / b # exp(-tms / decay_ms)
    return decay_ms

#SIM_ROOT = '/store/hbp/ich033/llandsme/simulations/'
#ENTROPY_DIR = '/store/hbp/ich033/llandsme/analysis/entropy'
#LOG_FILE = '/store/hbp/ich033/llandsme/analysis.data'

def main():
    SIM_ROOT = '/scratch/snx3000/llandsme/simulations/'
    ENTROPY_DIR = '/scratch/snx3000/llandsme/analysis/entropy/'
    LOG_FILE = '/scratch/snx3000/llandsme/analysis.data'

    sims = os.listdir(SIM_ROOT)

    for sim in sims:
        try:
            key = sim.split('.')[0]
            sim_fn = os.path.join(SIM_ROOT, sim)
            ent_fn = os.path.join(ENTROPY_DIR, sim)
            if os.path.exists(ent_fn):
                continue
            f = np.load(sim_fn)
            vs = np.array(f['vs'])
            t = np.array(f['t'])
            key = str(f['key'])
            sim_data = json.loads(base64.urlsafe_b64decode(key))
            first_spike = int(sorted(sim_data['spikes'])[0])
            S = get_entropy(vs)
            decay_ms = fit_curve(t, S, first_spike)
            np.savez_compressed(ent_fn, key=key, S=S, t=t)
            with open(LOG_FILE, 'a') as flog:
                print(f'{key} decay1ms {decay_ms}', file=flog)
            print(f'{ent_fn} decay1ms {decay_ms}')
        except Exception as ex:
            print(ex)

if __name__ == '__main__':
    main()
