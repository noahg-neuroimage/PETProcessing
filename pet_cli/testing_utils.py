import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import linregress


_TEXT_BOX_ = {'facecolor': 'lightblue', 'edgecolor': 'black', 'lw': 2.0, 'alpha': 0.2}


class TACPlots(object):
    r"""
    A class for plotting Time Activity Curves (TACs) on linear and semi-logarithmic scales.

    This class simplifies the process of comparing TACs on different scales. It generates a
    side-by-side plot with a linear-linear scale for the first plot and a log-x scale for the
    second plot. Users can add TACs to the plots and optionally generate a legend.

    Attributes:
        fig (matplotlib.figure.Figure): The figure object that contains the plots.
        axes (ndarray of Axes): The axes objects where the TACs are plotted.

    Example:
    
    .. code-block:: python

        tac_plots = TACPlots()
        tac_plots.add_tac(tac_times, tac_vals, label='TAC 1', pl_kwargs={'color': 'blue'})
        tac_plots.add_tac(tac_times_2, tac_vals_2, label='TAC 2', pl_kwargs={'color': 'red'})
        tac_plots.gen_legend()
        plt.show()

    """
    def __init__(self,
                 figsize: tuple = (8, 4),
                 xlabel: str = r'$t$ [minutes]',
                 ylabel: str = r'TAC [$\mathrm{kBq/ml}$]'):
        r"""
        Initialize the TACPlots with two subplots, one with a linear scale and the other with a semi-logarithmic scale.

        Args:
            figsize (tuple): The total size of the figure. Defaults to an 8x4 inches figure.
            xlabel (str): The label for the x-axis. Defaults to '$t$ [minutes]'.
            ylabel (str): The label for the y-axis. Defaults to 'TAC [$\mathrm{kBq/ml}$]'.
        """
        self.fig, self.axes = plt.subplots(1, 2, sharey=True, constrained_layout=True, figsize=figsize)
        self.fax = self.axes.flatten()
        [ax.set(xlabel=xlabel) for ax in self.fax]
        self.fax[0].set(ylabel=ylabel, title='Linear')
        self.fax[1].set(xscale='log', title='SemiLog-X')
    
    def add_tac(self, tac_times: np.ndarray, tac_vals: np.ndarray, label: str = None, pl_kwargs: dict = None):
        r"""
        Add a TAC to both subplots.

        Args:
            tac_times (np.ndarray): The time points for the TAC.
            tac_vals (np.ndarray): The corresponding values for the TAC.
            label (str): The label for the TAC in the legend. Defaults to no label.
            pl_kwargs (dict): Additional keyword arguments for the plot() function. Defaults to an empty dictionary.
        """
        if pl_kwargs is None:
            pl_kwargs = {'lw': 2, 'alpha': 0.8}
        if label is None:
            [ax.plot(tac_times, tac_vals, **pl_kwargs) for ax in self.fax]
        else:
            [ax.plot(tac_times, tac_vals, label=label, **pl_kwargs) for ax in self.fax]
    
    def gen_legend(self):
        r"""
        Generate a legend using the labels provided in the add_tac() method.

        Note:
            It is recommended to add all TACs before generating the legend. Any TACs added after the legend is
        generated will not be included in the legend.
        
        """
        self.fig.legend(*self.fax[0].get_legend_handles_labels(), bbox_to_anchor=(1.0, 0.5), loc='center left')


def generate_random_parameter_samples(num_samples, num_params, hi, lo):
    r"""
    Generates an array of random parameter samples.

    Args:
        num_samples (int): The number of samples to generate.
        num_params (int): The number of parameters per sample.
        hi (float): The upper limit (exclusive) of the interval to draw random samples.
        lo (float): The lower limit (inclusive) of the interval to draw random samples.

    Returns:
        np.ndarray: An array of random parameters samples with shape ``(num_samples, num_params)``.

    Note:
        Uses :func:`np.random.random` to generate random values in a given shape.
    """
    rand_samples = np.zeros((num_samples, num_params))
    if isinstance(hi, tuple):
        assert len(hi) == num_params, "`hi` must be of length num_params"
        if isinstance(lo, tuple):
            assert len(lo) == num_params, "`lo` must be of length num_params"
            for i in range(num_params):
                rand_samples[:, i] = np.random.random(num_samples) * (hi[i] - lo[i]) + lo[i]
        else:
            for i in range(num_params):
                rand_samples[:, i] = np.random.random(num_samples) * (hi[i] - lo) + lo
    elif isinstance(lo, tuple):
        assert len(lo) == num_params, "`lo` must be of length num_params"
        if isinstance(hi, tuple):
            assert len(hi) == num_params, "`hi` must be of length num_params"
            for i in range(num_params):
                rand_samples[:, i] = np.random.random(num_samples) * (hi[i] - lo[i]) + lo[i]
        else:
            for i in range(num_params):
                rand_samples[:, i] = np.random.random(num_samples) * (hi - lo[i]) + lo[i]
    else:
        rand_samples = np.random.random((num_samples, num_params)) * (hi - lo) + lo
    
    return rand_samples


def add_gaussian_noise_to_tac_based_on_max(tac_vals: np.ndarray, scale: float = 0.05) -> np.ndarray:
    r"""
    Adds Gaussian noise to a Time Activity Curve (TAC) based on the maximum TAC value.

    Args:
        tac_vals (np.ndarray): The Time Activity Curve values to which noise will be added.
        scale (float, optional): Scale of the noise to create. Defaults to 0.05.

    Returns:
        np.ndarray: Time Activity Curve with added Gaussian noise.

    Note:
        Uses :func:`np.random.normal` to generate random Gaussian noise.
        
    """
    noise = np.random.normal(loc=0.0, scale=np.max(tac_vals) * scale, size=tac_vals.shape)
    tac_out = tac_vals + noise
    tac_out[tac_out < 0] = 0.0
    tac_out[0] = 0.0
    return tac_out


def scatter_with_regression_figure(axes,
                                   fit_values: np.ndarray,
                                   true_values: np.ndarray,
                                   ax_titles: list[str],
                                   sca_kwargs: dict = None,
                                   reg_kwargs: dict = None):
    r"""
    Generates a scatter plot with regression for the specified axes, fitted and true values.
    
    The function is intended to be used to generate multiple sets of figures for comparing multiple fit parameters. The
    format of the values is ``values=[[first_param, second_param, ...], [first_param, second_param, ...], ..]``.

    Args:
        axes (list[matplotlib.axes]): The axes on which to generate the scatter plot.
        fit_values (np.ndarray): The fitted values to plot.
        true_values (np.ndarray): The true values to plot.
        ax_titles (list[str]): Titles of the axes.
        sca_kwargs (dict, optional): Additional keyword arguments for the scatter plot. Defaults to red dots.
        reg_kwargs (dict, optional): Additional keyword arguments for the regression line. Defaults to a black, solid line.

    Note:
        Uses :func:`scipy.stats.linregress` for linear regression.
    """
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, marker='.', color='red')
    
    if reg_kwargs is None:
        reg_kwargs = dict(ms=10, color='black', alpha=0.8, lw=3, ls='-')
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(true_values.T, fit_values.T, ax_titles))):
        x = xAr[~np.isnan(yAr)]
        y = yAr[~np.isnan(yAr)]
        fax[ax_id].scatter(x, y, **sca_kwargs)
        
        lin_reg = linregress(x, y)
        fax[ax_id].plot(x, x * lin_reg.slope + lin_reg.intercept, **reg_kwargs)
        
        fax[ax_id].text(0.05, 0.95, fr"$r^2={lin_reg.rvalue:<5.3f}$",
                        fontsize=20, transform=fax[ax_id].transAxes,
                        ha='left', va='top', bbox=_TEXT_BOX_)
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'True Values', ylabel=fr'Fit Values')


def bland_atlman_figure(axes,
                        fit_values: np.ndarray,
                        true_values: np.ndarray,
                        ax_titles: list[str],
                        sca_kwargs: dict = None,
                        bland_kwargs: dict = None):
    r"""
    Generates a Bland-Altman plot for the specified axes, fitted and true values.
    
    The function is intended to be used to generate multiple sets of figures for comparing multiple fit parameters. The
    format of the values is ``values=[[first_param, second_param, ...], [first_param, second_param, ...], ..]``.

    Args:
        axes (list[matplotlib.axes]): The axes on which to generate the Bland-Altman plot.
        fit_values (np.ndarray): The fitted values to plot.
        true_values (np.ndarray): The true values to plot.
        ax_titles (list[str]): Titles of the axes.
        sca_kwargs (dict, optional): Additional keyword arguments for the scatter plot. Defaults to red dots.
        bland_kwargs (dict, optional): Additional keyword arguments for Bland-Altman lines. Defaults to red lines.

    Note:
        Bland-Altman plot is a method of data plotting used in analyzing the agreement between two different assays.
    """
    
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, marker='.', color='red')
    
    if bland_kwargs is None:
        bland_kwargs = dict(s=10, color='red', alpha=0.8, lw=1)
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(fit_values.T, true_values.T, ax_titles))):
        x = (xAr + yAr) / 2.0
        y = xAr - yAr
        
        fax[ax_id].scatter(x, y, **sca_kwargs)
        
        mean_diff = np.nanmean(y)
        std_dev = np.nanstd(y)
        mid = mean_diff
        hi = mean_diff + 1.96 * std_dev
        lo = mean_diff - 1.96 * std_dev
        fax[ax_id].axhline(hi, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(lo, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(mid, ls='-', zorder=0, color=bland_kwargs['color'])
        
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'$\frac{{S_1+S_2}}{{2}}$ (Mean)', ylabel=fr'$S_1-S_2$ (Diff.)')


def ratio_bland_atlman_figure(axes,
                              fit_values: np.ndarray,
                              true_values: np.ndarray,
                              ax_titles: list[str],
                              sca_kwargs: dict = None,
                              bland_kwargs: dict = None):
    r"""
    Generates a ratio Bland-Altman plot for the specified axes, fitted and true values.

    The function is intended to be used to generate multiple sets of figures for comparing multiple fit parameters. The
    format of the values is ``values=[[first_param, second_param, ...], [first_param, second_param, ...], ..]``.

    Args:
        axes (list[matplotlib.axes]): The axes on which to generate the ratio Bland-Altman plot.
        fit_values (np.ndarray): The fitted values to plot.
        true_values (np.ndarray): The true values to plot.
        ax_titles (list[str]): Titles of the axes.
        sca_kwargs (dict, optional): Additional keyword arguments for the scatter plot. Defaults to red dots.
        bland_kwargs (dict, optional): Additional keyword arguments for ratio Bland-Altman lines. Defaults to a red.

    Note:
        A ratio Bland-Altman plot considers ratios, instead of differences, between the two methods; useful when
        the difference between the measures depends on the size of the measurement.
    """
    
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, marker='.', color='red')
    
    if bland_kwargs is None:
        bland_kwargs = dict(s=10, color='red', alpha=0.8, lw=1)
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(fit_values.T, true_values.T, ax_titles))):
        x = (np.log(xAr) + np.log(yAr)) / 2.0
        y = np.log(xAr) - np.log(yAr)
        
        fax[ax_id].scatter(x, y, **sca_kwargs)
        
        mean_diff = np.nanmean(y)
        std_dev = np.nanstd(y)
        mid = mean_diff
        hi = mean_diff + 1.96 * std_dev
        lo = mean_diff - 1.96 * std_dev
        
        fax[ax_id].axhline(hi, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(lo, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(mid, ls='-', zorder=0, color=bland_kwargs['color'])
        
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'$\frac{{\log S_1+\log S_2}}{{2}}$ (Mean)', ylabel=fr'$\log S_1-\log S_2$ (Diff.)')
