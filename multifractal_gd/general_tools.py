import numpy as np
from scipy.special import gamma

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def squarify(fig):
    """Make figure square.

    Args:
        fig (Figure): Figure for plot.
    """
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)


def mittag_leffler(z, a):
    """Mittag-Leffler function, E_{a,b}(z), for 0 < a < 1 and b = 0, 

    Args:
        z (float): Argument of Mittag-Leffler function.
        a (float): Order of Mittag-Leffler function.

    Returns:
        float: Value of E_{a, 0}(z).
    """
    z = np.atleast_1d(z)
    if a == 0:
        return 1/(1 - z)
    elif a == 1:
        return np.exp(z)
    elif a > 1 or all(z > 0):
        k = np.arange(100)
        return np.polynomial.polynomial.polyval(z, 1/gamma(a*k + 1))


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colors

    Args:
        xs (list): list of x coordinates
        ys (list): list of y coordinates
        c (list): list of numbers mapped to colormap
        ax (optional): Axes to plot on.
        kwargs (optional): passed to LineCollection

    Returns:
        LineCollection : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc
