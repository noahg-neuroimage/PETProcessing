"""
Simple module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils.time_activity_curve import TimeActivityCurve, MultiTACAnalysisMixin

def tacs_to_df(tacs_dir: str,
               participant: str):
    """
    Convert the TACs located in a folder into a pandas DataFrame.

    Args:
        tacs_dir (str): Path to directory containing TAC tsv files. Assumes TACs written by PETPAL
            functions, with one column for timing information, one column for mean regional
            activity.
        participant (str): Name of study participant the TAC belongs to.

    Returns:
        tacs (pd.DataFrame): pandas DataFrame containing timing info and mean activity for each 
            TAC in the TACs directory.
    """
    tacs_list = glob.glob(f'{tacs_dir}*')
    region_names = [os.path.basename(tac_file)[len('seg-'):-8] for tac_file in tacs_list]
    tacs = pd.DataFrame()
    reference_times = pd.read_csv(tacs_list[0],sep='\t')['FrameReferenceTime']
    for region in region_names:
        tac_file = pd.read_csv(f'{tacs_dir}/seg-{region}_tac.tsv',sep='\t')
        tac_file = tac_file.rename(columns={f'{region}_mean_activity': 'MeanActivity'})
        tac_file['Participant'] = participant
        tac_file['Region'] = region
        tac_file = tac_file.set_index('FrameReferenceTime')
        tac_file = tac_file.reindex(labels=reference_times,method='nearest')
        tac_file = tac_file.reset_index()
        tac_file['FrameReferenceTime'] = tac_file['FrameReferenceTime']/60
        tacs = pd.concat([tacs,tac_file])
    tacs = tacs.reset_index()
    return tacs


def tac_plots(tacs_data: pd.DataFrame,
              regions_to_plot: list):
    """
    Plot the TACs stored in a DataFrame, plotting only listed regions.
    Returns a seaborn plot object.

    Args:
        tacs_data (pd.DataFrame): DataFrame containing TAC data with three columns:
            FrameReferenceTime, MeanActivity, and Region.
        regions_to_plot (list): List of regions to be plotted in the TAC plot.
    
    Returns:
        tacs_plot (sns.Figure): Seaborn figure with lineplot of TACs for each included region.
    """
    tacs_to_plot = pd.DataFrame()
    for region in regions_to_plot:
        region_tac = tacs_data[tacs_data['Region'] == region]
        tacs_to_plot = pd.concat([tacs_to_plot,region_tac])
    tacs_plot = sns.lineplot(data=tacs_to_plot,x='FrameReferenceTime',y='MeanActivity',hue='Region',marker='o')
    tacs_plot.set_ylim(0,None)
    return tacs_plot


class TacFigure:
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

        tac_plots = TacFigure()
        tac_plots.add_tac(tac_times_in_minutes, tac_vals, label='TAC 1', color='blue')
        tac_plots.add_tac(tac_times_2, tac_vals_2, label='TAC 2', color='red')
        tac_plots.gen_legend()
        plt.show()

    """
    def __init__(self,
                 figsize: tuple = (8, 4),
                 xlabel: str = r'$t$ [minutes]',
                 ylabel: str = r'TAC [$\mathrm{kBq/ml}$]'):
        r"""
        Initialize the TacFigure with two subplots, one with a linear scale and the other with a
        semi-logarithmic scale.

        Args:
            figsize (tuple): The total size of the figure. Defaults to an 8x4 inches figure.
            xlabel (str): The label for the x-axis. Defaults to '$t$ [minutes]'.
            ylabel (str): The label for the y-axis. Defaults to 'TAC [$\mathrm{kBq/ml}$]'.
        """
        self.fig, self.axes = self.setup_linear_and_log_subplot(figsize=figsize)
        self.fax = self.axes.flatten()
        _xlabel_set = [ax.set(xlabel=xlabel) for ax in self.fax]
        self.fax[0].set(ylabel=ylabel, title='Linear')
        self.fax[1].set(xscale='log', title='SemiLog-X')


    def setup_linear_and_log_subplot(self, figsize: tuple):
        """
        Get the figure and axes objects for a 1x2 MatPlotLib subplot.

        Args:
            figsize (tuple): Size of the figure.
        """
        return plt.subplots(1, 2, sharey=True, constrained_layout=True, figsize=figsize)


    def add_tac(self, tac_times: np.ndarray, tac_vals: np.ndarray, **kwargs):
        r"""
        Add a TAC to both subplots.

        Args:
            tac_times (np.ndarray): The time points for the TAC.
            tac_vals (np.ndarray): The corresponding values for the TAC.
            kwargs (dict): Additional keyword arguments for the plot() function.
        """
        return [ax.plot(tac_times, tac_vals, **kwargs) for ax in self.fax]


    def add_errorbar(self,
                     tac_times: np.ndarray,
                     tac_vals: np.ndarray,
                     uncertainty: np.ndarray,
                     **kwargs):
        """
        Add errorbars to a TAC plot.

        Args:

        """
        return [ax.errorbar(tac_times, tac_vals, yerr=uncertainty, **kwargs) for ax in self.fax]

    def gen_legend(self):
        r"""
        Generate a legend using the labels provided in the add_tac() method.

        Note:
            It is recommended to add all TACs before generating the legend. Any TACs added after
        the legend is generated will not be included in the legend.

        """
        handles, labels = self.fax[0].get_legend_handles_labels()
        if handles:
            self.fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.5), loc='center left')


class RegionalTacFigure(TacFigure,MultiTACAnalysisMixin):
    """
    Handle plotting regional TACs generated with PETPAL.
    """
    def __init__(self,
                 tacs_dir: str, 
                 figsize: tuple = (8, 4),
                 xlabel: str = r'$t$ [minutes]',
                 ylabel: str = r'TAC [$\mathrm{kBq/ml}$]'):
        MultiTACAnalysisMixin.__init__(self,input_tac_path='',tacs_dir=tacs_dir)
        TacFigure.__init__(self,figsize=figsize,xlabel=xlabel,ylabel=ylabel)
        self.figure = sns.lineplot()

    @property
    def tacs_objects_list(self):
        return self.get_tacs_objects_list_from_files_list(self.tacs_files_list)


    def get_figure(self):
        return self.figure.get_figure()


    def plot_tac(self,tac: TimeActivityCurve):
        """
        Plot a single TAC from the TimeActivityCurve object.
        """
        self.figure = sns.lineplot(x=tac.times,y=tac.activity,ax=self.figure)
        return self.figure


    def plot_tac_errorbar(self,tac: TimeActivityCurve):
        """
        Plot a single TAC from the TimeActivityCurve object with errorbars.
        """
        self.plot_tac(tac=tac)
        self.figure.errorbar(x=tac.times,y=tac.activity,yerr=tac.uncertainty)
        return self.figure


    def plot_regional_tacs(self):
        """
        Placeholder
        """
        tacs_obj_list = self.tacs_objects_list()
        for tac in tacs_obj_list:
            self.plot_tac_errorbar(tac=tac)
        return self.get_figure()


    def plot_tacs_in_regions_list(self,regions: list[str | int]):
        """
        Placeholder
        """
        tacs_obj_list = self.tacs_objects_list()
        for region in regions:
            tac = tacs_obj_list[region]  # NOTE: THIS DOESNT DO ANYTHING
            self.plot_tac_errorbar(tac=tac)
        return self.get_figure()
