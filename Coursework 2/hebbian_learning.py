import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz 

def corr_input(T=100000, rho=.8, angle=1/12):
    """
    creates a 2 x T matrix containing the time-series of a 2D signal,
    with the two components correlated.
    :param T: duration of the signals (in time-steps)
    :param rho: correlation coefficient of the two components.
    :param angle: angle of rotation of the 2D signal (in units of full revolution) relative to horizontal.

    :return: 2D array X of shape (2, T)
    """
    angle = 2 * np.pi * (angle - 1 / 8)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mix = np.array([[np.sqrt(1 - rho**2), rho], [0, 1]])
    X = rot @ mix @ npr.randn(2, T)

    return X

def plot_pc_lines(v, ax, col_v='k', col_perp='k', **kwargs):
    """
    plots two perpendicular lines passing through the origin,
    and spanning the area bounded by the axis frame.
    One line (solid) is along the v, the other (dashed) perpendicular to it.
    :param v: a 2D vector determining the direction of the first line
    :param ax: pyplot axis to draw on
    :param col_v: color of the first line
    :param col_perp: color of the 2nd (perpendicular) line
    :param kwargs: optional keyword arguments passed to the plt.plot function
    """
    if v[0] == 0:
        ax.axvline(0, '-', color=col_v, **kwargs, zorder=5)
        ax.axhline(0, '--',  color=col_perp, **kwargs, zorder=5)
    elif v[1] == 0:
        ax.axhline(0, '-', color=col_v, **kwargs, zorder=5)
        ax.axvline(0, '--', color=col_perp, **kwargs, zorder=5)
    else:
        slope = v[1] / v[0]
        slope2 = - 1 / slope
        
        xlims = 5 * np.array(ax.get_xlim())
        ylims = 5 * np.array(ax.get_ylim())
        
        # Calculate intersection points with axes' frame
        if np.abs(slope) <= np.diff(ylims) / np.diff(xlims):
            # Line along v intersects frame on x-axis bounds
            ax.plot(xlims, slope * xlims, '-', color=col_v, **kwargs, zorder=5)
            ax.plot(ylims / slope2, ylims, '--', color=col_perp, **kwargs, zorder=5)
        else:
            # Line along v intersects frame on x-axis bounds
            ax.plot(ylims / slope, ylims, '-', color=col_v, **kwargs, zorder=5)
            ax.plot(xlims, slope2 * xlims, '--', color=col_perp, **kwargs, zorder=5)
        
def scatter(X, pc1=None, ax=None, c='tab:blue',
            alpha=.2, figsize=np.array((8,6))*.6, lw=1, ms=4):
    """
    Makes a scatter plot of the two components of signals contained in X against each other.
    If pc1 is provided, also draws lines along the provided vector `pc1` and perpendicular
    to it (see help for `plot_pc_lines`).
    :param X: 2D array of shape (2, T)
    :param pc1: 2D vector (1D array-like of shape (2,)) intended to represent the direction of the 1st PC
                of the data X.
    :param ax: a pyplot axis to draw on -- if not provided an axis will be created.
    :param c: color of scatter plot points (could be a list of length T, giving different color to different points)
    :param alpha: opacity of scatter plot points
    :param figsize: figure size
    :param lw: linewidth for PC1/2 lines
    :param ms: size of scatter plot points
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X[0], X[1], c=c, alpha=alpha, s=ms, zorder=5)
    ax.set_aspect(1)
    ax.set_box_aspect(1)
    if pc1 is not None:
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
        plot_pc_lines(pc1, ax, lw=lw)
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)

def normalize_W(W):
    """
    normalizes the rows of the matrix W to have unit length (L2 or Euclidean norm)
    :param W: 2D array of shape (N, M)

    :return: normalized W with same shape
    """
    return W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))

def plot_results(X, w, pc1=None, skip_step=1, figscale=.5, ms=10, lw_arrow=6):
    """
    Makes a 2-panel plot to visualise the learning trajectory and final outcome of Hebbian learning of w.
    Right panel: will show the final 2D weight-vector w[-1] (this is a row of the weight matrix W, hence the weights are
    those targeting one of the post-synaptic or output-layer neurons) as a red vector, which is superimposed on
    a scatter plot of input during training.
    Left panel: plots the learning history of the weights w over time in the 2D plane (the color of the points
    reflects time; the later in the history the brighter the color). This is superimposed by lines depicting the PC
    directions of the input data. (Since weights are normalized by construction, the unit circle is also drawn.)

    :param X: input timeseries -- 2D array of shape (2, T)
    :param w: learning trajectory of weights onto one postsynaptic neuron: 2D array of shape (T, 2).
              This could be a slice of the W_history 3D array such as W_history[:, 4, :] yielding the learning
              history of the weights onto the 4th postsynaptic neuron.
    :param pc1: 2D vector (1D array-like of shape (2,)) intended to represent the direction of the 1st PC
                of the data X.
    :param skip_step: only every `skip_step` time-steps of weight history will be plotted in left panel.
    :param figscale: figure scale
    :param ms: size of scatter plot points
    :param lw_arrow: linewidth for the red arrow representing w[-1]

    return: the figure and its axes.
    """
    fig, axs = plt.subplots(1, 2, figsize=np.array((8 * 2, 6)) * figscale)
    ax = axs[1]
    scatter(X, ax=ax)
    length = .4
    ax.quiver(0, 0, w[-1, 0], w[-1, 1], scale=1/length, lw=lw_arrow, color='red', zorder=5)

    ax = axs[0]
    w = w[::skip_step]
    ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), 'k', lw=0.5, zorder=5)
    scatter(w.T, ax=ax, pc1=pc1, c=np.arange(w.shape[0]), ms=ms, alpha=.9)
    ax.set_yticks(ax.get_xticks())
    xlim = ylim = np.array([-1, 1]) * 1.2
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, axs
    
def make_K(N, l1=.1, l2=.2, ratio=.5, a=1):
    """
    Makes the spatial linear response kernel of the inter-connected output layer.
    For the default parameters, the impulse response of the output layer  (a column of K)
    has a Mexican hat (MH) profile.
    More generally, the profile is a difference (or sum, if ratio is negative) of Gaussians.
    :param N: number of output neurons
    :param l1: length-scale of the positive lobe of the MH.
    :param l2: length-scale of the negative lobe of the MH.
    :param ratio: ralative strengh of negative to positive lobes.
    :param a: overall strength of K

    :return: K, a (symmetric and Toeplitz) square matrix -- or in Python: a 2D array of shape (N, N)
    """
    assert N > 1
    pos = np.linspace(0, 1, N + 1)[:-1]
    K_prof = a * (np.exp(- pos**2 / 2 / l1**2) - ratio * np.exp(- pos**2 / 2 / l2**2))
    return toeplitz(K_prof) #, pos