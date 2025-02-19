import numpy as np 
from scipy.special import gamma as G
from fbm import fbm

def integrate_overdamped_FLE(x0, T, eta, H, beta, gradient, seed=None):
    if seed is not None:
        np.random.seed(seed)
    alpha = 2-2*H
    coeff_1 = eta/(G(2*H-1) * G(3-2*H))
    coeff_2 = eta * np.sqrt(2 / (beta * G(3-2*H) * G(2*H-1)) )
    B = fbm(n=T, hurst=1-H, length=T, method='daviesharte')
    
    gradient_history = np.zeros(T)
    history_weights = np.arange(T,-1,-1)
    history_weights = np.power(history_weights[:-1], alpha) - np.power(history_weights[1:], alpha)
    
    trajectory = np.zeros(T+1)
    trajectory[0] = x0
    for j in range(1,T+1):
        gradient_history[j-1] = gradient(trajectory[j-1])
        sum_term = np.dot(gradient_history[:j], history_weights[T-j:])
        trajectory[j] = x0 - coeff_1 * sum_term - coeff_2 * B[j]
    return trajectory

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
                d += (positions[tw+i] - positions[tw+i+t])**2
            d /= windowsize
            msd.append(d)
        xs.append(x)
        ys.append(msd)
    return xs, ys