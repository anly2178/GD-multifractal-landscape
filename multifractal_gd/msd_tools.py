import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multifractal_gd.general_tools import multiline


def calculate_TAMSD(positions, waiting_times, tau, windowsize):
    """Calculates time-average mean squared displacement (TAMSD).

    Args:
        positions (ndarray): Positions of trajectory.
        waiting_times (list): Waiting times.
        tau (list): Largest lag time.
        windowsize (int): Window for time-average.

    Returns:
        list: Each element of the xs is a list of x coordinates 
                corresponding to a waiting time.
        list: Each element of the ys is a list of y coordinates 
                corresponding to a waiting time.
    """
    n = len(waiting_times)
    xs = []
    ys = []
    for j in range(n):
        tw = waiting_times[j]
        msd = []
        x = list(range(tau[j]))
        for t in x:
            d = 0
            for i in range(windowsize):
                d += np.linalg.norm(positions[tw+i, :] -
                                    positions[tw+i+t, :])**2
            d /= windowsize
            msd.append(d)
        xs.append(x)
        ys.append(msd)
    return xs, ys


def plot_TAMSD(xs, ys, waiting_times, save_path=None):
    """Quick function for plotting the TAMSD.

    Args:
        xs (list): x values.
        ys (list): y values.
        waiting_times (list): Waiting times.
        save_path (str): Path for saving figure. Defaults to None, in which case not saved.
    """
    fig1, ax1 = plt.subplots()
    lc = multiline(xs, ys, waiting_times, cmap='jet')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\tau$ (iterations)')
    ax1.set_ylabel('TAMSD')
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in', which='both')
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    axcb = fig1.colorbar(lc, pad=0.01, format=fmt)
    axcb.formatter.set_powerlimits((0, 0))
    axcb.set_label(r'$t_w$ (iterations)')
    lc.set_clim(vmin=0, vmax=max(waiting_times))

    BIGGER_SIZE = 14

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    ax1.set_aspect(1./ax1.get_data_ratio(), adjustable='box')
    plt.show()
    if isinstance(save_path, str):
        fig1.savefig(save_path, dpi=600, bbox_inches='tight')
    else:
        return
