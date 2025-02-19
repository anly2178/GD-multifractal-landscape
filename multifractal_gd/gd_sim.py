import warnings
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def simulate_GD(landscape, T, lr, xystart, save_results=True, save_path=None):
    """Simulates gradient descent on a landscape with reflecting boundary conditions. 
    Can save dictionary of results in the same folder as the landscape under fname with gradient, trajectory, and plotting segments.

    Args:
        landscape (str or ndarray): Path to file for landscape or ndarray of landscape.
        T (int): Total number of iterations
        lr (float): Learning rate. 
        xystart (tup or list): Initial (x,y) position. 
        save_results (bool): If True, dictionary of results are saved to fpath under fname. If False, dictionary of results are returned.
            Defaults to True.
        save_path (str): Path for saving. Defaults to None.

    Returns:
        dict:   "landscape" contains landscape
                "trajectory" contains [x,y] position of optimiser at each iteration
                "segments" contains segments for plotting purposes only
                "trajectory_continuous" contains [x,y] position accounting for reflecting boundary conditions for calculation purposes.
    """
    if isinstance(landscape, str):
        # Load landscape
        landscape = loadmat(landscape)["landscape"]
    assert (type(landscape) is np.ndarray), "landscape is not a numpy array"

    M = landscape.shape[0]
    grad_landscape = np.gradient(landscape)
    xystart = np.array(xystart, dtype='float64')

    # Variables to keep track of, particularly for plotting
    coords = np.zeros((T, 2))
    coords[0, :] = xystart
    coords_cont = np.zeros((T, 2))
    coords_cont[0, :] = xystart
    niters = 1
    segments = []  # segments for plotting because of BCs
    x = [xystart[0]]
    y = [xystart[1]]
    xyold = xystart
    xynew = np.zeros(2)
    xy_cont = xystart
    rbcs = np.array([1, 1])

    grad = []

    # Simulation
    while niters < T:
        xygrad = _interpolate_grad(xyold, grad_landscape)
        grad.append(xygrad)
        xynew = xyold - xygrad * lr
        xy_cont = xy_cont - np.multiply(xygrad, rbcs) * lr
        if any(xynew // np.array([M-1, M-1])):
            if len(x) != 0:
                segments.append([x, y, False])
            for i in range(len(xynew)):
                x_or_y = copy.deepcopy(xynew[i])
                while x_or_y // (M-1) != 0:
                    if x_or_y < 0:
                        x_or_y *= -1
                    elif x_or_y > 0:
                        excess = x_or_y - (M-1)
                        x_or_y = (M-1) - excess
                xynew[i] = x_or_y
        x.append(xynew[0])
        y.append(xynew[1])
        coords[niters, :] = xynew
        coords_cont[niters, :] = xy_cont
        xyold = xynew
        niters += 1

    if len(x) != 0:
        segments.append([x, y, False])

    mdic = {"landscape": landscape, "trajectory": coords,
            "segments": segments, "trajectory_continuous": coords_cont}
    if save_results:
        assert isinstance(save_path, str)
        savemat(save_path, mdic)
    return mdic


def plot_trajectory(results, linecolour='k', cmap=cm.terrain, sfcolour='r', mincolour='m', legend_loc='best', include_min=True, return_figax=False, savepath=None):
    """Plots trajectory on fractal landscape. If return_figax is True, then figure and axes are returned.

    Args:
        results (str or dict): Path to file containing dictionary results of simulation or dictionary itself.
        linecolour (str, list, optional): Colour of trajectory line. Defaults to 'k' (black). Alternatively,
            if linecolour is a list then colours are used corresponding to the regimes. E.g., ['r', 'g', 'b', ...]
        cmap (cmap, optional): Colour map from matplotlib. Defaults to cm.terrain.
        sfcolour (str, optional): Colour of markers for start and finish positions. Defaults to 'r'.
        mincolour (str, optional): Colour of marker for global minimum position. Defaults to 'm'.
        include_min (bool, optional): Whether to plot the global minimum with a marker. Defaults to True.
        return_figax (bool, optional): Whether to return the figure and axis. Defaults to False.
        savepath (str, optional): Path for saving. Defaults to None.
    """
    if isinstance(results, str):
        results = loadmat(results)
    assert isinstance(results, dict)

    landscape = results["landscape"]
    traj = results["trajectory"]
    segments = results["segments"]

    fig, ax = _plot_trajectory(
        landscape=landscape, segments=segments, cmap=cmap, linecolour=linecolour)

    # Plot markers
    ax.scatter(traj[0, 0], traj[0, 1], s=50, c=sfcolour,
               marker='^', label='Start', zorder=2, rasterized=True)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=50, c=sfcolour,
               marker='v', label='Finish', zorder=2, rasterized=True)
    if include_min:
        ind = np.where(landscape == np.min(landscape))
        ax.scatter(ind[1], ind[0], s=50, c=mincolour,
                   marker='*', label='Global minimum', zorder=2, rasterized=True)

    ax.legend(loc=legend_loc)
    if type(savepath) == str:
        fig.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print("Figure not saved as savepath was not given.")
    plt.show()
    if return_figax:
        return fig, ax

def _interpolate_grad(coord, grad):
    """Calculates 2D linear interpolation from a surface.

    Args:
        coord (array): (x,y) position
        grad (list of ndarrays): output from np.gradient of fractal landscape
    """
    x = coord[0]
    y = coord[1]
    if (int(x) == x) and (int(y) == y):
        return np.array([grad[1][int(y),int(x)], grad[0][int(y),int(x)]])
    else:
        f11_x = grad[1][int(np.floor(y)), int(np.floor(x))]; f11_y = grad[0][int(np.floor(y)), int(np.floor(x))]
        f21_x = grad[1][int(np.floor(y)), int(np.ceil(x))]; f21_y = grad[0][int(np.floor(y)), int(np.ceil(x))]
        f12_x = grad[1][int(np.ceil(y)), int(np.floor(x))]; f12_y = grad[0][int(np.ceil(y)), int(np.floor(x))]
        f22_x = grad[1][int(np.ceil(y)), int(np.ceil(x))]; f22_y = grad[0][int(np.ceil(y)), int(np.ceil(x))]
        fval_x = f11_x*(np.ceil(x)-x)*(np.ceil(y)-y) + f21_x*(x-np.floor(x))*(np.ceil(y)-y) + f12_x*(np.ceil(x)-x)*(y-np.floor(y)) + f22_x*(x-np.floor(x))*(y-np.floor(y))
        fval_y = f11_y*(np.ceil(x)-x)*(np.ceil(y)-y) + f21_y*(x-np.floor(x))*(np.ceil(y)-y) + f12_y*(np.ceil(x)-x)*(y-np.floor(y)) + f22_y*(x-np.floor(x))*(y-np.floor(y))
        return np.array([fval_x, fval_y])

def _plot_trajectory(landscape, segments, cmap, linecolour):
    fig, ax = plt.subplots()
    ax.imshow(landscape, cmap=cmap, origin='lower', rasterized=True)
    # Plot trajectory with bold line
    for elem in segments:
        if not elem[-1]:
            ax.plot(elem[0].flatten(), elem[1].flatten(),
                    color=linecolour, linewidth=1, zorder=1)
        else:
            ax.plot(elem[0][0][:2].flatten(), elem[1][0]
                    [:2].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
            ax.plot(elem[0][0][2:4].flatten(), elem[1][0]
                    [2:4].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
        if len(elem[0][0]) == 6:
            ax.plot(elem[0][0][4:6].flatten(), elem[1][0]
                    [4:6].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
    return fig, ax