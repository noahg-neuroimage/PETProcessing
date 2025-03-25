"""
Simple module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import glob
import argparse
import os
import pandas as pd
import seaborn as sns

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
