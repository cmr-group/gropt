import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, floor
import gropt


def get2_eddy_mode0(G, lam, dt):
    E0 = np.zeros_like(G)
    for i in range(G.size):
        ii = float(G.size - i - 1)
        if i == 0:
            val = -np.exp(-ii*dt/lam)
        else:
            val = np.exp(-(ii+1.0)*dt/lam) - np.exp(-ii*dt/lam)
        E0[i] = -val
    
    return E0

def get2_eddy_mode1(G, lam, dt):
    E1 = np.zeros_like(G)
    
    val = 0.0
    for i in range(G.size):
        ii = float(G.size - i - 1)
        val += -np.exp(-ii*dt/lam)
    E1[0] = val * 1e3 * dt
    
    for i in range(1, G.size):
        ii = float(G.size - i)
        val = -np.exp(-ii*dt/lam)
        E1[i] = val  * 1e3 * dt
    
    return E1


def get_eddy_curves(G, dt, max_lam, n_lam):
    all_lam = np.linspace(1e-4, max_lam, n_lam)
    all_e0 = []
    all_e1 = []
    for lam in all_lam:
        lam = lam * 1.0e-3

        E0 = get2_eddy_mode0(G, lam, dt)
        all_e0.append(np.sum(E0*G))

        E1 = get2_eddy_mode1(G, lam, dt)
        all_e1.append(np.sum(E1*G))

    return all_lam, all_e0, all_e1

def get_min_TE(params, bval = 1000, min_TE = -1, max_TE = -1, verbose = 0):
    if params['mode'][:4] == 'diff':
        if min_TE < 0:
            min_TE = params['T_readout'] + params['T_90'] + params['T_180'] + 10
        
        if max_TE < 0:
            max_TE = 200

        G_out, T_out = get_min_TE_diff(params, bval, min_TE, max_TE, verbose)

    elif params['mode'] == 'free':
        if min_TE < 0:
            min_TE = 0.1
        
        if max_TE < 0:
            max_TE = 5.0

        G_out, T_out = get_min_TE_free(params, min_TE, max_TE, verbose)
    
    return G_out, T_out


def get_min_TE_diff(params, target_bval, min_TE, max_TE, verbose = 0):
    
    T_lo = min_TE
    T_hi = max_TE
    T_range = T_hi-T_lo

    best_time = 999999.9

    if 'dt' in params:
        dt = params['dt']
    else:
        dt = 1.0e-3/params['N0']

    if verbose:
        print('Testing TE =', end='', flush=True)
    while ((T_range*1e-3) > (dt/4.0)): 
        params['TE'] = T_lo + (T_range)/2.0
        if verbose:
            print(' %.3f' % params['TE'], end='', flush=True)
        G, ddebug = gropt.gropt(params, verbose)
        lim_break = ddebug[14]
        bval = get_bval(G, params)
        if bval > target_bval:
            T_hi = params['TE']
            if T_hi < best_time:
                G_out = G
                T_out = T_hi
                best_time = T_hi
        else:
            T_lo = params['TE']
        T_range = T_hi-T_lo

    if verbose:
        print(' Final TE = %.3f ms' % T_out)

    params['TE'] = T_out
    return G_out, T_out

def get_min_TE_free(params, min_TE, max_TE, verbose = 0):
    
    T_lo = min_TE
    T_hi = max_TE
    T_range = T_hi-T_lo

    best_time = 999999.9

    if 'dt' in params:
        dt = params['dt']
    else:
        dt = 1.0e-3/params['N0']

    if verbose:
        print('Testing TE =', end='', flush=True)
    while ((T_range*1e-3) > (dt/4.0)): 
        params['TE'] = T_lo + (T_range)/2.0
        if verbose:
            print(' %.3f' % params['TE'], end='', flush=True)
        G, ddebug = gropt.gropt(params)
        lim_break = ddebug[14]
        if lim_break == 0:
            T_hi = params['TE']
            if T_hi < best_time:
                G_out = G
                T_out = T_hi
                best_time = T_hi
        else:
            T_lo = params['TE']
        T_range = T_hi-T_lo

    if verbose:
        print(' Final TE = %.3f ms' % T_out)

    return G_out, T_out



def get_stim(G, dt):
    alpha = 0.333
    r = 23.4
    c = 334e-6
    Smin = r/alpha
    coeff = []
    for i in range(G.size):
        coeff.append( c / ((c + dt*(G.size-1) - dt*i)**2.0) / Smin )
    coeff = np.array(coeff)

    stim_out = []
    for j in range(G.size-1):
        ss = 0
        for i in range(j+1):
            ss += coeff[coeff.size-1-j+i]*(G[i+1]-G[i])
        stim_out.append(ss)

    stim_out = np.array(stim_out)
    return stim_out

def get_moments(G, T_readout, dt):
    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    scaler = np.zeros(Nm)
    for mm in range(Nm):
        tMat[mm] = tvec**mm
        scaler[mm] = (dt*1e3)**mm
                                 
    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    return moments

def get_bval(G, params):

    if params['dt'] < 0:
        dt = (params['TE']-params['T_readout']) * 1.0e-3 / G.size
    else:
        dt = params['dt']
    
    TE = G.size*dt*1.0e3 + params['T_readout']
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    
    INV = np.ones(G.size)
    INV[tINV:] = -1
    
    Gt = 0
    bval = 0
    for i in range(G.size):
        if i < tINV:
            Gt += G[i] * dt
        else:
            Gt -= G[i] * dt
        bval += Gt*Gt*dt

    bval *= (GAMMA*2*np.pi)**2
    
    return bval

def plot_moments(G, T_readout, dt):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]

    plt.figure()
    mmt = np.cumsum(mm[0])*1e3
    plt.plot(mmt/np.abs(mmt).max())
    mmt = np.cumsum(mm[1])*1e6
    plt.plot(mmt/np.abs(mmt).max())
    mmt = np.cumsum(mm[2])*1e9
    plt.plot(mmt/np.abs(mmt).max())
    plt.axhline(0, color='k')
    
    
def plot_moments_actual(G, T_readout, dt):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = dt*tMat@(G*INV)
    mm = dt*tMat * (G*INV)[np.newaxis,:]
    
    return mm

def get_moment_plots(G, T_readout, dt, diffmode = 1):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    #GAMMA = 42.58e3; 
    INV = np.ones(G.size)
    if diffmode > 0:
        INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    #moments = np.abs(GAMMA*dt*tMat@(G*INV))
    #mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]
    moments = dt*tMat@(G*INV)
    mm = dt*tMat * (G*INV)[np.newaxis,:]

    out = []
    for i in range(Nm):
        mmt = np.cumsum(mm[i])
        out.append(mmt)

    return out

def get_moment_plots_orig(G, T_readout, dt, diffmode = 1):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3;     # units: 1/(mT x ms) 
    INV = np.ones(G.size)
    if diffmode > 0:
        INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt     # units: sec
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]     # units: (1/(mT x ms))x(sec)x(mT/m)...seems wrong
                                                   # units: (sec)x(mT/m)...seems wrong
    out = []
    for i in range(Nm):
        mmt = np.cumsum(mm[i])
        out.append(mmt)

    return out

def get_moment_plots_arfi(G, T_readout, dt):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 2
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = dt*tMat@(G*INV)
    mm = dt*tMat * (G*INV)[np.newaxis,:]

    out = []
    for i in range(Nm):
        mmt = np.cumsum(mm[i])
        out.append(mmt)

    return out, INV


def plot_waveform_new(G, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True, plot_maxwell = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    TE = params['TE']
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    tt = np.arange(G.size) * dt * 1e3
    tINV = TE/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1
    if plot_maxwell: 
        N_plots += 1
        
    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
        
    axarr[i_row, i_col].plot(tt, G*1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, dt, diffmode)                       
    if plot_moments:
        for i in range(3):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt/np.abs(mmt).max())
        axarr[i_row, i_col].set_ylabel('$M_{n}$ [AU]')        
        axarr[i_row, i_col].set_title('Moments')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_maxwell:
        GAM = 267.52e3 # [rad/(mT x sec)]
        tINV = int(np.floor(TE/params['dt']/1.0e3/2.0))
        INV = np.ones(G.size)
        INV[tINV:] = -1
        tt = np.arange(G.size)*params['dt']*1e3
        conPhase = np.cumsum(1000*INV*(G**2)*GAM*dt/1.0e3)
        axarr[i_row, i_col].plot(tt,conPhase)
        axarr[i_row, i_col].set_title('Concominant Phase')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('Maxwell Index [AU]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1            
            
    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return bval, axarr


def plot_waveform(G, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    TE = params['TE']
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    #dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt * 1e3
    tINV = TE/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
        
    #if diffmode > 1:
    #    axarr[i_row, i_col].axvline(tINV, linestyle='--', color='0.7')
    axarr[i_row, i_col].plot(tt, G*1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    
    axarr[i_row, i_col].set_ylim([-55,55])
    axarr[i_row, i_col].set_xlim([0,120])
    
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, dt, diffmode)                       
    if plot_moments:
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        for i in range(3):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt/np.abs(mmt).max())
        axarr[i_row, i_col].set_ylabel('$M_{n}$ [AU]')        
        axarr[i_row, i_col].set_title('Moments')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        
        axarr[i_row, i_col].set_xlim([0,120])
        axarr[i_row, i_col].set_ylim([-1.1,1.1])        
        
        
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return bval


def plot_waveform_arfi(G, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    TE = params['TE']
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    #dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt * 1e3
    tINV = TE/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
        
    #if diffmode > 1:
    #    axarr[i_row, i_col].axvline(tINV, linestyle='--', color='0.7')
    axarr[i_row, i_col].plot(tt, G*1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    
    axarr[i_row, i_col].set_ylim([-55,55])
    axarr[i_row, i_col].set_xlim([0,120])
    
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, dt, diffmode)                       
    if plot_moments:
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        for i in range(3):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt/np.abs(mmt).max())
        axarr[i_row, i_col].set_ylabel('$M_{n}$ [AU]')        
        axarr[i_row, i_col].set_title('Moments')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        
        axarr[i_row, i_col].set_xlim([0,120])
        axarr[i_row, i_col].set_ylim([-1.1,1.1])        
        
        
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return bval, axarr

    
def plot_waveform_overlap(G, G_, TE, TE_, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    #dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt * 1e3
    tt_ = np.arange(G_.size) * dt * 1e3
    tINV = TE/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
        
    #if diffmode > 1:
    #    axarr[i_row, i_col].axvline(tINV, linestyle='--', color='0.7')
    axarr[i_row, i_col].plot(tt, G*1000, color='b')
    axarr[i_row, i_col].plot(tt_, G_*1000, color='r')
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt, color='b')
        axarr[i_row, i_col].plot(tt_[:-1], np.diff(G_)/dt, color='r')        
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, dt, diffmode)                       
    if plot_moments:
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        for i in range(3):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt/np.abs(mmt).max())
        axarr[i_row, i_col].set_ylabel('$M_{n}$ [AU]')        
        axarr[i_row, i_col].set_title('Moments')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))
        pns_ = np.abs(get_stim(G_, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color='k')
        
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns, color='b')
        axarr[i_row, i_col].plot(tt_[:-1], pns_, color='r')
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return bval, axarr


def plot_waveform_psd(G, dt, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    
    TE = G.size * dt * 1e3
    T_readout = 0
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    axarr[i_row, i_col].plot(tt, G)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        
        axarr[i_row, i_col].set_title('Slew Rate')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('SR [mT/m/ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size, 0)
    if plot_moments:
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        Nm=1
        for i in range(Nm):
            mmt = mm[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt, mmt)
            if i == 1:
                axarr[i_row, i_col].plot(tt, mmt)
            if i == 2:
                axarr[i_row, i_col].plot(tt, mmt)
        
        axarr[i_row, i_col].set_title('Moment')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('M$_{0}$ [mT/m x ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy')
        axarr[i_row, i_col].set_xlabel('lam [ms]')
    #     axarr[i_row, i_col].set_ylabel(' [AU]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95]) 
    
    return mm, axarr

    
def plot_waveform_simple(G, dt, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    
    TE = G.size * dt * 1e3
    T_readout = 0
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    axarr[i_row, i_col].plot(tt, G)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        
        axarr[i_row, i_col].set_title('Slew Rate')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('SR [mT/m/ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size, 0)
    if plot_moments:
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        Nm=1
        for i in range(Nm):
            mmt = mm[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt, mmt)
            if i == 1:
                axarr[i_row, i_col].plot(tt, mmt)
            if i == 2:
                axarr[i_row, i_col].plot(tt, mmt)
        
        axarr[i_row, i_col].set_title('Moment')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('M$_{0}$ [mT/m x ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy')
        axarr[i_row, i_col].set_xlabel('lam [ms]')
    #     axarr[i_row, i_col].set_ylabel(' [AU]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95]) 
    
    return mm

def plot_waveform_simple_overlap(G, G_, dt, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    
    TE = G.size * dt * 1e3
    T_readout = 0
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt
    tt_ = np.arange(G_.size) * dt*1000
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    axarr[i_row, i_col].plot(tt, G, color='b')
    axarr[i_row, i_col].plot(tt_, G_, color='r')
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt, color='b')
        axarr[i_row, i_col].plot(tt_[:-1], np.diff(G_)/(dt*1000), color='r')
        
        axarr[i_row, i_col].set_title('Slew Rate')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('SR [mT/m/ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, dt, 0)
    mm_ = get_moment_plots(G_, T_readout, dt*1000, 0)
    if plot_moments:
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        Nm=1
        for i in range(Nm):
            mmt = mm[i]
            mmt_ = mm_[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt, mmt, color='b')
                axarr[i_row, i_col].plot(tt_, mmt_, color='r')                
            if i == 1:
                axarr[i_row, i_col].plot(tt, mmt)
                axarr[i_row, i_col].plot(tt_, mmt_)                
            if i == 2:
                axarr[i_row, i_col].plot(tt, mmt)
                axarr[i_row, i_col].plot(tt_, mmt_)                
        
        axarr[i_row, i_col].set_title('Moment')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('M$_{0}$ [mT/m x ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy')
        axarr[i_row, i_col].set_xlabel('lam [ms]')
    #     axarr[i_row, i_col].set_ylabel(' [AU]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1


    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95]) 
    
    return mm, axarr

def plot_waveform_flow(G, dt, Nm, plot_moments = True, plot_slew = True):
    sns.set()
    sns.set_context("talk")
    
    TE = G.size * dt * 1e6
    T_readout = 0
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    axarr[i_row, i_col].plot(tt, G)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('SR [mT/m/ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size, 0)            
            
    if plot_moments:
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        for i in range(Nm):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt)
            axarr[i_row, i_col].set_ylabel('$M_{0}$ [mT/m x ms]')
            if Nm > 1:
                #axarr[i_row, i_col].plot(tt, mmt)
                axarr[i_row, i_col].set_ylabel('$M_{n}$ [mT/m x ms$^{n+1}$]')        
        axarr[i_row, i_col].set_title('Moment(s)')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95]) 
    
    return mm, axarr


def plot_waveform_EC(G, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    TE = params['TE']
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt * 1e3
    tINV = TE/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
        
    axarr[i_row, i_col].plot(tt, G*1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size, 0)            
            
    if plot_moments:
        for i in range(2):
            mmt = mm[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt, mmt*1e6)
            if i == 1:
                axarr[i_row, i_col].plot(tt, mmt*1e6)
            if i == 2:
                axarr[i_row, i_col].plot(tt, mmt*1e9)
            axarr[i_row, i_col].set_ylabel('$M_{n}$ [mT/m x ms$^{n+1}$]')        
        axarr[i_row, i_col].set_title('Moment(s)')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].legend(('$M_{0}$', '$M_{1}$','$M_{2}$'),prop={'size': 10},labelspacing=-0.5,loc=0)

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
            all_e.append(100*r@G)
        
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return mm, axarr

def plot_waveform_flow_EC(G1, G2, params, plot_moments = True, plot_eddy = True, plot_pns = True, plot_slew = True,
                  suptitle = '', eddy_lines=[], eddy_range = [1e-3,120,1000]):
    sns.set()
    sns.set_context("talk")
    dt = params['dt'];
    
    TE1 = params['TE1']
    T_readout1 = params['T_readout1']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    dt1 = (TE1-T_readout1) * 1.0e-3 / G1.size
    tt1 = np.arange(G1.size) * dt1 * 1e3
    tINV1 = TE1/2.0
    
    TE2 = params['TE2']
    T_readout2 = params['T_readout2']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    dt2 = (TE2-T_readout2) * 1.0e-3 / G2.size
    tt2 = np.arange(G2.size) * dt2 * 1e3
    tINV2 = TE2/2.0
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_eddy: 
        N_plots += 1
    if plot_pns: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    bval1 = get_bval(G1, params)
    bval2 = get_bval(G2, params)
        
    axarr[i_row, i_col].plot(tt1, G1*1000)
    axarr[i_row, i_col].plot(tt2, G2*1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt1[:-1], np.diff(G1)/dt1)
        axarr[i_row, i_col].plot(tt2[:-1], np.diff(G2)/dt2)        
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    mm1 = get_moment_plots(G1*1000, T_readout1, (TE1-T_readout1) * 1.0e-3 / G1.size, 0)            
    mm2 = get_moment_plots(G2*1000, T_readout2, (TE2-T_readout2) * 1.0e-3 / G2.size, 0)            
    
    if plot_moments:
        for i in range(3):
            mmt1 = mm1[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt1, mmt1*1e3)
            if i == 1:
                axarr[i_row, i_col].plot(tt1, mmt1*1e6)
            if i == 2:
                axarr[i_row, i_col].plot(tt1, mmt1*1e9)
            axarr[i_row, i_col].set_ylabel('$M_{n}$ [mT/m x ms$^{n+1}$]')        
            
        for i in range(3):
            mmt2 = mm2[i]
            if i == 0:
                axarr[i_row, i_col].plot(tt2, mmt2*1e3)
            if i == 1:
                axarr[i_row, i_col].plot(tt2, mmt2*1e6)
            if i == 2:
                axarr[i_row, i_col].plot(tt2, mmt2*1e9)
            axarr[i_row, i_col].set_ylabel('$M_{n}$ [mT/m x ms$^{n+1}$]')        
            
        axarr[i_row, i_col].set_title('Moment(s)')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].legend(('$M_{0}$', '$M_{1}$','$M_{2}$'),prop={'size': 10},labelspacing=-0.5,loc=0)

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0],eddy_range[1],eddy_range[2])
        all_e1 = []
        all_e2 = []        
        for lam in all_lam:
            lam = lam * 1.0e-3
            r1 = np.diff(np.exp(-np.arange(G1.size+1)*dt1/lam))[::-1]
            all_e1.append(100*r1@G1)
            r2 = np.diff(np.exp(-np.arange(G2.size+1)*dt2/lam))[::-1]
            all_e2.append(100*r2@G2)
        
        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        
        axarr[i_row, i_col].plot(all_lam, all_e1)
        axarr[i_row, i_col].plot(all_lam, all_e2)
        axarr[i_row, i_col].plot(all_lam, np.abs(np.subtract(all_e1,all_e2)))        
        axarr[i_row, i_col].set_title('Eddy Currents')
        axarr[i_row, i_col].set_xlabel('$\lambda$ [ms]')
        axarr[i_row, i_col].legend(('$EC_{1}$', '$EC_{2}$','$|EC_{1}-EC_{2}|$'),prop={'size': 10},labelspacing=-0.1,loc=0)
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns1 = np.abs(get_stim(G1, dt1))
        pns2 = np.abs(get_stim(G2, dt2))
        
        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(tt1[:-1], pns1)
        axarr[i_row, i_col].plot(tt2[:-1], pns2)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])
    
    return mm1, mm2, axarr
    
def plot_waveform_flow_overlap(G, G_, dt, Nm, plot_moments = True, plot_slew = True):
    sns.set()
    sns.set_context("talk")
    
    TE = G.size * dt * 1e6
    T_readout = 0
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt
    tt_ = np.arange(G_.size) * dt
    
    N_plots = 1
    if plot_moments: 
        N_plots += 1
    if plot_slew: 
        N_plots += 1

    N_rows = 1 + (N_plots-1)//3
    N_cols = ceil(N_plots/N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows*3.5))
    
    i_row = 0
    i_col = 0

    axarr[i_row, i_col].plot(tt, G, color='b')
    axarr[i_row, i_col].plot(tt_, G_, color='r')
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('Time [ms]')
    axarr[i_row, i_col].set_ylabel('G [mT/m]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        axarr[i_row, i_col].plot(tt[:-1], np.diff(G)/dt)
        
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].set_ylabel('SR [mT/m/ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_moments:
        mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size, 0)
        #axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        for i in range(Nm):
            mmt = mm[i]
            axarr[i_row, i_col].plot(tt, mmt)
            axarr[i_row, i_col].set_ylabel('$M_{0}$ [mT/m x ms]')
            if Nm > 1:
                #axarr[i_row, i_col].plot(tt, mmt)
                axarr[i_row, i_col].set_ylabel('$M_{n}$ [mT/m x ms$^{n+1}$]')        
        axarr[i_row, i_col].set_title('Moment(s)')
        axarr[i_row, i_col].set_xlabel('Time [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 0.5, 0.95]) 
    
    return axarr    
    
def conventional_triangles(SR_Max, M0, dt):
    
    r = np.sqrt((abs(M0))/SR_Max)
    h = np.sqrt((abs(M0)*SR_Max))
    r_ = int(np.ceil(r/dt)) # round up to nearest integer multiple of dt
    h_ = (M0/r_)/dt
    Tri = np.concatenate((np.linspace(0,h_,r_),np.linspace(h_,0,r_)),axis=0)
    
    return Tri

def conventional_trapezoids(G_Max, SR_Max, M0, dt):
    
    h = G_Max
    r = abs(h)/SR_Max
    p = abs(M0/h)-r
    r_ = int(np.ceil(r/dt)) # round up to nearest integer multiple of dt
    p_ = int(np.ceil(p/dt)) # round up to nearest integer multiple of dt
    h_ = (M0/(p_+r_))/dt
    Trap = np.concatenate((np.linspace(0,h_,r_),np.linspace(h_,h_,p_),np.linspace(h_,0,r_)),axis=0)
    
    return Trap    

def conventional_flowcomp(params):
    
    G_ss = np.concatenate((np.linspace(params['g_ss']*1e-3,params['g_ss']*1e-3,int(ceil(params['p_ss']*1e-3/params['dt']))),np.linspace(params['g_ss']*1e-3,params['g_ss']*1e-3/10,10)),axis=0)
    t_ss = G_ss.size*params['dt']
    mmt = get_moment_plots(G_ss, 0, params['dt'], 0)
    M0S = mmt[0][-1]*1e6
    M1S = mmt[1][-1]*1e9
    M2S = mmt[2][-1]*1e12
    
    ramp_range = np.linspace(1e-03,5,5000)
    for r in ramp_range:
        r_ = int(ceil(r/params['dt']/1000))
        h = r*params['smax']
        M02 = (-h*r+np.sqrt((h*r)**2+2*(h*r*M0S + M0S**2 + 2*h*M1S)))/2
        M01 = M02 + M0S
        w1 = M01/h + r
        w2 = M02/h + r
        w1_ = int(ceil(w1/params['dt']/1000))
        w2_ = int(ceil(w2/params['dt']/1000))
        
        if  (w1_-2*r_ <= 1) or (w2_-2*r_ <= 1):
            h1 = M01/(r_-w1_)*100
            h2 = -M02/(r_-w2_)*100
            break
        
    G = np.concatenate((np.linspace(0,h1,r_),np.linspace(h1,h1,w1_-2*r_),np.linspace(h1,h2,2*r_),np.linspace(h2,h2,w2_-2*r_),np.linspace(h2,0,r_)),axis=0)
    FC = np.concatenate((G_ss*1000,G),axis=0)
    
    return FC, M0S, M1S, M2S, t_ss, G_ss

def slice_select_bridge(params):
    
    G_ss = np.linspace(params['g_ss']*1e-3,params['g_ss']*1e-3,int(ceil(params['p_ss']*1e-3/params['dt'])))
    t_ss = G_ss.size*params['dt']
    mmt = get_moment_plots(G_ss, 0, params['dt'], 0)
    M0S = mmt[0][-1]*1e6
    M1S = mmt[1][-1]*1e9
    M2S = mmt[2][-1]*1e12
    
    return M0S, M1S, M2S, t_ss, G_ss

def conventional_flowencode(params):

    GAM = 2*np.pi*42.57         # 1/(s*T)
    DeltaM1 = np.pi/(GAM*params['VENC'])  # mT/m * ms^2

    ramp_range = np.linspace(0.01,0.5,491)
    for r in ramp_range:
        params['h'] = r*params['smax']
        M02 = np.abs((-params['h']*r+np.sqrt((params['h']*r)**2+2*(params['h']*r*params['M0S'] + params['M0S']**2 + 2*params['h']*(params['M1S']+DeltaM1))))/2) 
        M01 = params['M0S'] + M02 
        w1 = np.abs(M01)/params['h'] + r 
        w2 = np.abs(M02)/params['h'] + r 
        r_ = int(ceil(r/params['dt']/1000))
        w1_ = int(ceil(w1/params['dt']/1000))
        w2_ = int(ceil(w2/params['dt']/1000))
        
        if  (w1_-2*r_ <= 1) or (w2_-2*r_ <= 1):
            break
        
    G = np.concatenate((np.linspace(0,-params['h'],r_),np.linspace(-params['h'],-params['h'],w1_-2*r_),np.linspace(-params['h'],0,r_),np.linspace(params['h']/r_,params['h'],r_-1),np.linspace(params['h'],params['h'],w2_-2*r_),np.linspace(params['h'],0,r_)),axis=0)
    FE = np.concatenate((params['G_ss']*1000,G),axis=0)
    
    return FE, DeltaM1

def monopolar_diffusion(params):
    
    params['dt'] = 1e-5
    h = params['gmax']/1000
    SR_Max = params['smax']/1000
    GAM = 2*np.pi*42.58e3
    zeta = (h/SR_Max)*1e-3
    Delta = np.real((zeta**2/12 + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**2/(4*GAM**4*h**4))/((- (zeta**2/12 + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**2/(4*GAM**4*h**4))**3 + ((GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**3/(8*GAM**6*h**6) - (- 3*params['b'] - GAM**2*params['T_90']**3*h**2 + GAM**2*params['T_180']**3*h**2 + GAM**2*params['T_readout']**3*h**2 + (8*GAM**2*h**2*zeta**3)/5 - 3*GAM**2*params['T_90']*params['T_180']**2*h**2 + 3*GAM**2*params['T_90']**2*params['T_180']*h**2 - 3*GAM**2*params['T_90']*params['T_readout']**2*h**2 + 3*GAM**2*params['T_90']**2*params['T_readout']*h**2 + 3*GAM**2*params['T_180']*params['T_readout']**2*h**2 + 3*GAM**2*params['T_180']**2*params['T_readout']*h**2 - (7*GAM**2*params['T_90']*h**2*zeta**2)/2 + 3*GAM**2*params['T_90']**2*h**2*zeta + (7*GAM**2*params['T_180']*h**2*zeta**2)/2 + 3*GAM**2*params['T_180']**2*h**2*zeta + (7*GAM**2*params['T_readout']*h**2*zeta**2)/2 + 3*GAM**2*params['T_readout']**2*h**2*zeta - 6*GAM**2*params['T_90']*params['T_180']*params['T_readout']*h**2 - 6*GAM**2*params['T_90']*params['T_180']*h**2*zeta - 6*GAM**2*params['T_90']*params['T_readout']*h**2*zeta + 6*GAM**2*params['T_180']*params['T_readout']*h**2*zeta)/(4*GAM**2*h**2) + (zeta**2*(GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta))/(16*GAM**2*h**2))**2)**(1/2) + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**3/(8*GAM**6*h**6) - (- 3*params['b'] - GAM**2*params['T_90']**3*h**2 + GAM**2*params['T_180']**3*h**2 + GAM**2*params['T_readout']**3*h**2 + (8*GAM**2*h**2*zeta**3)/5 - 3*GAM**2*params['T_90']*params['T_180']**2*h**2 + 3*GAM**2*params['T_90']**2*params['T_180']*h**2 - 3*GAM**2*params['T_90']*params['T_readout']**2*h**2 + 3*GAM**2*params['T_90']**2*params['T_readout']*h**2 + 3*GAM**2*params['T_180']*params['T_readout']**2*h**2 + 3*GAM**2*params['T_180']**2*params['T_readout']*h**2 - (7*GAM**2*params['T_90']*h**2*zeta**2)/2 + 3*GAM**2*params['T_90']**2*h**2*zeta + (7*GAM**2*params['T_180']*h**2*zeta**2)/2 + 3*GAM**2*params['T_180']**2*h**2*zeta + (7*GAM**2*params['T_readout']*h**2*zeta**2)/2 + 3*GAM**2*params['T_readout']**2*h**2*zeta - 6*GAM**2*params['T_90']*params['T_180']*params['T_readout']*h**2 - 6*GAM**2*params['T_90']*params['T_180']*h**2*zeta - 6*GAM**2*params['T_90']*params['T_readout']*h**2*zeta + 6*GAM**2*params['T_180']*params['T_readout']*h**2*zeta)/(4*GAM**2*h**2) + (zeta**2*(GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta))/(16*GAM**2*h**2))**(1/3) + ((((GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**3/(8*GAM**6*h**6) - (3*(- (GAM**2*params['T_90']**3*h**2)/3 + GAM**2*params['T_90']**2*params['T_180']*h**2 + GAM**2*params['T_90']**2*params['T_readout']*h**2 + GAM**2*params['T_90']**2*h**2*zeta - GAM**2*params['T_90']*params['T_180']**2*h**2 - 2*GAM**2*params['T_90']*params['T_180']*params['T_readout']*h**2 - 2*GAM**2*params['T_90']*params['T_180']*h**2*zeta - GAM**2*params['T_90']*params['T_readout']**2*h**2 - 2*GAM**2*params['T_90']*params['T_readout']*h**2*zeta - (7*GAM**2*params['T_90']*h**2*zeta**2)/6 + (GAM**2*params['T_180']**3*h**2)/3 + GAM**2*params['T_180']**2*params['T_readout']*h**2 + GAM**2*params['T_180']**2*h**2*zeta + GAM**2*params['T_180']*params['T_readout']**2*h**2 + 2*GAM**2*params['T_180']*params['T_readout']*h**2*zeta + (7*GAM**2*params['T_180']*h**2*zeta**2)/6 + (GAM**2*params['T_readout']**3*h**2)/3 + GAM**2*params['T_readout']**2*h**2*zeta + (7*GAM**2*params['T_readout']*h**2*zeta**2)/6 + (8*GAM**2*h**2*zeta**3)/15 - params['b']))/(4*GAM**2*h**2) + (zeta**2*(GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta))/(16*GAM**2*h**2))**2 - (zeta**2/12 + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**2/(4*GAM**4*h**4))**3)**(1/2) + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)**3/(8*GAM**6*h**6) - (3*(- (GAM**2*params['T_90']**3*h**2)/3 + GAM**2*params['T_90']**2*params['T_180']*h**2 + GAM**2*params['T_90']**2*params['T_readout']*h**2 + GAM**2*params['T_90']**2*h**2*zeta - GAM**2*params['T_90']*params['T_180']**2*h**2 - 2*GAM**2*params['T_90']*params['T_180']*params['T_readout']*h**2 - 2*GAM**2*params['T_90']*params['T_180']*h**2*zeta - GAM**2*params['T_90']*params['T_readout']**2*h**2 - 2*GAM**2*params['T_90']*params['T_readout']*h**2*zeta - (7*GAM**2*params['T_90']*h**2*zeta**2)/6 + (GAM**2*params['T_180']**3*h**2)/3 + GAM**2*params['T_180']**2*params['T_readout']*h**2 + GAM**2*params['T_180']**2*h**2*zeta + GAM**2*params['T_180']*params['T_readout']**2*h**2 + 2*GAM**2*params['T_180']*params['T_readout']*h**2*zeta + (7*GAM**2*params['T_180']*h**2*zeta**2)/6 + (GAM**2*params['T_readout']**3*h**2)/3 + GAM**2*params['T_readout']**2*h**2*zeta + (7*GAM**2*params['T_readout']*h**2*zeta**2)/6 + (8*GAM**2*h**2*zeta**3)/15 - params['b']))/(4*GAM**2*h**2) + (zeta**2*(GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta))/(16*GAM**2*h**2))**(1/3) + (GAM**2*params['T_180']*h**2 - GAM**2*params['T_90']*h**2 + GAM**2*params['T_readout']*h**2 + GAM**2*h**2*zeta)/(2*GAM**2*h**2))
    delta = Delta + params['T_90'] - params['T_180'] - params['T_readout'] - zeta
    b = GAM**2*h**2*(delta**2*(Delta-delta/3) + zeta**3/30 - delta*zeta**2/6)
    T_90_ = int(ceil(params['T_90']/params['dt']))
    zeta_ = int(np.floor(zeta/params['dt']))
    delta_ = int(ceil((delta-2*zeta)/params['dt']))
    Delta_ = int(ceil((Delta-zeta-delta+params['T_180']/2)/params['dt']))
    Mono = np.concatenate((np.linspace(0,0,T_90_),
                           np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_),
                           np.linspace(0,0,Delta_),
                           np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_)))
#     T_90_ = int(1e5*params['T_90'])
#     zeta_ = int(1e5*zeta)
#     delta_ = int(1e5*(delta-2*zeta))
#     Delta_ = int(1e5*(Delta-zeta-delta+params['T_180']/2))
#     Mono = np.concatenate((np.linspace(0,0,T_90_),
#                            np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_),
#                            np.linspace(0,0,Delta_),
#                            np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_)))
    TE = Mono.size

    return Mono, TE, b

def bipolar_diffusion(params):
    
    params['dt'] = 1e-5
    h = params['gmax']/1000
    SR_Max = params['smax']/1000
    GAM = 2*np.pi*42.58e3
    zeta = (h/SR_Max)*1e-3
    delta = (((9*(params['b'] - (GAM**2*h**2*zeta**3)/15)**2)/(64*GAM**4*h**4) - zeta**6/1728)**(1/2) + (3*(params['b'] - (GAM**2*h**2*zeta**3)/15))/(8*GAM**2*h**2))**(1/3) + zeta**2/(12*(((9*(params['b'] - (GAM**2*h**2*zeta**3)/15)**2)/(64*GAM**4*h**4) - zeta**6/1728)**(1/2) + (3*params['b'] - (GAM**2*h**2*zeta**3)/5)/(8*GAM**2*h**2))**(1/3))
    b = (GAM**2*h**2*(20*delta**3 - 5*delta*zeta**2 + zeta**3))/15
#     T_90_ = int(ceil(params['T_90']/params['dt']))
#     T_180_ = int(ceil(params['T_180']/params['dt']))
#     zeta_ = int(np.floor(zeta/params['dt']))
#     delta_ = int(ceil((delta-2*zeta)/params['dt']))
#     gap = int(ceil((params['T_readout']-0.5*params['T_90'])/params['dt']))
    T_90_ = int(1e5*params['T_90'])
    T_180_ = int(1e5*params['T_180'])
    T_readout_ = int(1e5*params['T_readout'])
    zeta_ = int(1e5*zeta)
    delta_ = int(1e5*(delta-2*zeta))
    gap = T_readout_ - 0.5*T_90_
    Bipolar = np.concatenate((np.linspace(0,0,T_90_),
                              np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_),
                              np.linspace(0,-h,zeta_),np.linspace(-h,-h,delta_),np.linspace(-h,0,zeta_),
                              np.linspace(0,0,gap),
                              np.linspace(0,0,T_180_),
                              np.linspace(0,h,zeta_),np.linspace(h,h,delta_),np.linspace(h,0,zeta_),
                              np.linspace(0,-h,zeta_),np.linspace(-h,-h,delta_),np.linspace(-h,0,zeta_)))
    TE = Bipolar.size
    
    return Bipolar, TE, b

#def asymmbipolar_diffusion(params, delta1, Delta):
def asymmbipolar_diffusion(params):

    params['dt'] = 1e-5    
    h = params['gmax']/1000
    SR_Max = params['smax']/1000
    GAM = 2*np.pi*42.58e3
    zeta = (h/SR_Max)*1e-3

    # *** Fixed for now, need to solve the symbolic math equations *** 
    # These values are for the Optimization paper
    delta1 = 0.0135
    delta2 = 0.0210
    Delta = 0.0733
    
    delta2 = (delta1*(Delta-zeta))/(Delta-2*delta1+zeta)
    
    b = (GAM**2*h**2*(20*Delta**3*delta1**3 - 30*Delta**3*delta1**2*zeta - 5*Delta**3*delta1*zeta**2 + 16*Delta**3*zeta**3 - 60*Delta**2*delta1**4 + 120*Delta**2*delta1**3*zeta - 5*Delta**2*delta1**2*zeta**2 - 106*Delta**2*delta1*zeta**3 + 48*Delta**2*zeta**4 - 100*Delta*delta1**3*zeta**2 + 252*Delta*delta1**2*zeta**3 - 197*Delta*delta1*zeta**4 + 48*Delta*zeta**5 + 40*delta1**6 - 120*delta1**5*zeta + 200*delta1**4*zeta**2 - 268*delta1**3*zeta**3 + 227*delta1**2*zeta**4 - 96*delta1*zeta**5 + 16*zeta**6))/(15*(Delta - 2*delta1 + zeta)**3)
    T_90_ = int(1e5*params['T_90'])
    T_180_ = int(1e5*params['T_180'])
    T_readout_ = int(1e5*params['T_readout'])
    zeta_ = int(1e5*zeta)
    delta1_ = int(1e5*(delta1-2*zeta))
    delta2_ = int(1e5*(delta2-2*zeta))    
    gap = int(1e5*(Delta-delta1-delta2))
    AsymmBipolar = np.concatenate((np.linspace(0,0,T_90_),
                                   np.linspace(0,-h,zeta_),np.linspace(-h,-h,delta1_),np.linspace(-h,0,zeta_),
                                   np.linspace(0,h,zeta_),np.linspace(h,h,delta2_),np.linspace(h,0,zeta_),
                                   np.linspace(0,0,gap),
                                   np.linspace(0,h,zeta_),np.linspace(h,h,delta2_),np.linspace(h,0,zeta_),
                                   np.linspace(0,-h,zeta_),np.linspace(-h,-h,delta1_),np.linspace(-h,0,zeta_)),axis=0)
    TE = AsymmBipolar.size
    
    return AsymmBipolar, TE, b

def maxwell_analysis(G, params, Rot, tensor, B0, position, corr):
    
    #COLOR = 'white'
    #plt.rcParams['text.color'] = COLOR
    #plt.rcParams['axes.labelcolor'] = COLOR
    #plt.rcParams['xtick.color'] = COLOR
    #plt.rcParams['ytick.color'] = COLOR
    
    if corr == 0:
        f, axarr = plt.subplots(3, 3, squeeze=False, figsize=(12, 3*3.5))
    else:
        f, axarr = plt.subplots(4, 3, squeeze=False, figsize=(12, 4*3.5))            
    
    tt = np.arange(G.size) * params['dt'] * 1e3

    tINV = int(np.floor(params['TE']/params['dt']/1.0e3/2.0))
    INV = np.ones(G.size)
    INV[tINV:] = -1
    GAM = 42.58e3
    
    G = G*1000;
    Desired = np.outer(Rot*tensor,G)              # Desired gradient waveforms
    Concominant1 = np.zeros(shape=(1,len(G)))     # Concominant gradient matrix x
    Concominant2 = np.zeros(shape=(1,len(G)))     # Concominant gradient matrix y
    Concominant3 = np.zeros(shape=(1,len(G)))     # Concominant gradient matrix z
    Actual = np.zeros(shape=(3,len(G)))     
    Corrected = np.zeros(shape=(3,len(G)))     

    for i in range(len(G)):
        tmp1 = np.inner([[Desired[2][i]**2,0,-2*Desired[0][i]*Desired[2][i]],
                        [0,Desired[2][i]**2,-2*Desired[1][i]*Desired[2][i]],
                        [-2*Desired[0][i]*Desired[2][i],-2*Desired[1][i]*Desired[2][i],4*Desired[0][i]**2 + 4*Desired[1][i]**2]],(1/(4*B0)))
        tmp2 = np.reshape(np.inner(tmp1,position),(3,1))
        Concominant1[0][i] = tmp2[0]
        Concominant2[0][i] = tmp2[1]
        Concominant3[0][i] = tmp2[2]

    # Actual gradient waveforms    
    Actual[0][:] = Desired[0][:] + Concominant1                
    Actual[1][:] = Desired[1][:] + Concominant2
    Actual[2][:] = Desired[2][:] + Concominant3

    # Corrected gradient waveforms
    Corrected[0][:] = Actual[0][:] - Concominant1
    Corrected[1][:] = Actual[1][:] - Concominant2
    Corrected[2][:] = Actual[2][:] - Concominant3

    # Plot the Desired Gradients and Desired M0
    axarr[0,0].plot(tt,Desired[0][:])
    axarr[0,0].plot(tt,Desired[1][:])
    axarr[0,0].plot(tt,Desired[2][:])
    axarr[0,0].set_xlabel('Time [ms]')
    axarr[0,0].set_ylabel('G [mT/m]')
    axarr[0,0].set_title('Desired Waveforms')
    #axarr[0,0].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)
   
    axarr[0,1].plot(tt,GAM*np.cumsum(INV*(Desired[0][:])*params['dt']/1000))
    axarr[0,1].plot(tt,GAM*np.cumsum(INV*(Desired[1][:])*params['dt']/1000))
    axarr[0,1].plot(tt,GAM*np.cumsum(INV*(Desired[2][:])*params['dt']/1000))
    axarr[0,1].set_xlabel('Time [ms]')
    axarr[0,1].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[0,1].set_title('Desired Zeroth Moment')
    #axarr[0,1].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)

    axarr[0,2].plot(tt,GAM*np.cumsum(INV*(Desired[0][:])*params['dt']/1000))
    axarr[0,2].plot(tt,GAM*np.cumsum(INV*(Desired[1][:])*params['dt']/1000))
    axarr[0,2].plot(tt,GAM*np.cumsum(INV*(Desired[2][:])*params['dt']/1000))
    axarr[0,2].set_xlabel('Time [ms]')
    axarr[0,2].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[0,2].set_title('Desired Zeroth Moment')
    axarr[0,2].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=1)
    axarr[0,2].set_xlim(len(G)*params['dt']*1e3-1,len(G)*params['dt']*1e3+1)
    #axarr[0,2].set_ylim(-1e-3,1e-3)

    # Plot the Concominant Gradients and Concominant M0
    axarr[1,0].plot(tt,Concominant1.flatten())
    axarr[1,0].plot(tt,Concominant2.flatten())
    axarr[1,0].plot(tt,Concominant3.flatten())
    axarr[1,0].set_xlabel('Time [ms]')
    axarr[1,0].set_ylabel('G [mT/m]')
    axarr[1,0].set_title('Concominant Waveforms')
    #axarr[1,0].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)
   
    axarr[1,1].plot(tt,GAM*np.cumsum(INV*(Concominant1.flatten())*params['dt']/1000))
    axarr[1,1].plot(tt,GAM*np.cumsum(INV*(Concominant2.flatten())*params['dt']/1000))
    axarr[1,1].plot(tt,GAM*np.cumsum(INV*(Concominant3.flatten())*params['dt']/1000))
    axarr[1,1].set_xlabel('Time [ms]')
    axarr[1,1].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[1,1].set_title('Concominant Zeroth Moment')
    #axarr[1,1].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)

    axarr[1,2].plot(tt,GAM*np.cumsum(INV*(Concominant1.flatten())*params['dt']/1000))
    axarr[1,2].plot(tt,GAM*np.cumsum(INV*(Concominant2.flatten())*params['dt']/1000))
    axarr[1,2].plot(tt,GAM*np.cumsum(INV*(Concominant3.flatten())*params['dt']/1000))
    axarr[1,2].set_xlabel('Time [ms]')
    axarr[1,2].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[1,2].set_title('Concominant Zeroth Moment')
    #axarr[1,2].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=1)
    axarr[1,2].set_xlim(len(G)*params['dt']*1e3-1,len(G)*params['dt']*1e3+1)
    axarr[1,2].set_ylim(-1e-3,1e-3)
    
    # Plot the Actual Gradients and Actual M0
    axarr[2,0].plot(tt,Actual[0][:])
    axarr[2,0].plot(tt,Actual[1][:])
    axarr[2,0].plot(tt,Actual[2][:])
    axarr[2,0].set_xlabel('Time [ms]')
    axarr[2,0].set_ylabel('G [mT/m]')
    axarr[2,0].set_title('Actual Waveforms')
    #axarr[2,0].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)
   
    axarr[2,1].plot(tt,GAM*np.cumsum(INV*(Actual[0][:])*params['dt']/1000))
    axarr[2,1].plot(tt,GAM*np.cumsum(INV*(Actual[1][:])*params['dt']/1000))
    axarr[2,1].plot(tt,GAM*np.cumsum(INV*(Actual[2][:])*params['dt']/1000))
    axarr[2,1].set_xlabel('Time [ms]')
    axarr[2,1].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[2,1].set_title('Actual Zeroth Moment')
    #axarr[2,1].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)

    axarr[2,2].plot(tt,GAM*np.cumsum(INV*(Actual[0][:])*params['dt']/1000))
    axarr[2,2].plot(tt,GAM*np.cumsum(INV*(Actual[1][:])*params['dt']/1000))
    axarr[2,2].plot(tt,GAM*np.cumsum(INV*(Actual[2][:])*params['dt']/1000))
    axarr[2,2].set_xlabel('Time [ms]')
    axarr[2,2].set_ylabel('$M_{0}$ [mT/m x ms]')
    axarr[2,2].set_title('Actual Zeroth Moment')
    #axarr[2,2].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=1)
    axarr[2,2].set_xlim(len(G)*params['dt']*1e3-1,len(G)*params['dt']*1e3+1)
    axarr[2,2].set_ylim(-1e-3,1e-3)
    
    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1.5, 1]) 

    if corr == 1:
        # Plot the Corrected Gradients and Corrected M0
        axarr[3,0].plot(tt,Corrected[0][:])
        axarr[3,0].plot(tt,Corrected[1][:])
        axarr[3,0].plot(tt,Corrected[2][:])
        axarr[3,0].set_xlabel('Time [ms]')
        axarr[3,0].set_ylabel('G [mT/m]')
        axarr[3,0].set_title('Corrected Waveforms')
        #axarr[3,0].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)

        axarr[3,1].plot(tt,GAM*np.cumsum(INV*(Corrected[0][:])*params['dt']/1000))
        axarr[3,1].plot(tt,GAM*np.cumsum(INV*(Corrected[1][:])*params['dt']/1000))
        axarr[3,1].plot(tt,GAM*np.cumsum(INV*(Corrected[2][:])*params['dt']/1000))
        axarr[3,1].set_xlabel('Time [ms]')
        axarr[3,1].set_ylabel('$M_{0}$ [mT/m x ms]')
        axarr[3,1].set_title('Corrected Zeroth Moment')
        #axarr[3,1].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=2)

        axarr[3,2].plot(tt,GAM*np.cumsum(INV*(Corrected[0][:])*params['dt']/1000))
        axarr[3,2].plot(tt,GAM*np.cumsum(INV*(Corrected[1][:])*params['dt']/1000))
        axarr[3,2].plot(tt,GAM*np.cumsum(INV*(Corrected[2][:])*params['dt']/1000))
        axarr[3,2].set_xlabel('Time [ms]')
        axarr[3,2].set_ylabel('$M_{0}$ [mT/m x ms]')
        axarr[3,2].set_title('Corrected Zeroth Moment')
        #axarr[3,2].legend(('$G_{x}$', '$G_{y}$', '$G_{z}$'),prop={'size': 12},labelspacing=-0.1,loc=1)
        axarr[3,2].set_xlim(len(G)*params['dt']*1e3-1,len(G)*params['dt']*1e3+1)
        axarr[3,2].set_ylim(-1e-3,1e-3)

    
    return axarr    
