"""
Simple module for processing dosimetry data.
"""
import numpy as np
import pandas as pd
import datetime
from scipy.optimize import curve_fit
import seaborn as sns

def time_midpoint(
        start_times: list[float],
        frame_durations: list[float],
        half_life: float
        ) -> np.ndarray:
    """
    Compute the exponential midpoint frame time based on the isotope half life.
    """
    start_times_np = np.array(start_times)
    frame_durations_np = np.array(frame_durations)
    decay_constant = np.log(2) / half_life
    midlength = 1 / decay_constant * np.log(decay_constant*frame_durations_np / (1-np.exp(-decay_constant * frame_durations_np)))
    midpoints = start_times_np + midlength
    return midpoints


def organ_dose(
        activity_cps: list[float],
        organ_volume: list[float],
        calibration_factor: float,
        injected_dose: float,
        scale_factor: float
        ) -> np.ndarray:
    """
    Compute the dose to each organ as a fraction of the injected dose.
    """
    activity_array = []
    for i,_activity in enumerate(activity_cps):
        activity_array += [
            activity_cps.iloc[i]
                * organ_volume.iloc[i]
                * calibration_factor
                / injected_dose
                * scale_factor]
    return activity_array


def convert_frame_time_seconds(
        start_times: list[datetime.time],
        injection_time: datetime.time
        ) -> np.ndarray:
    """
    Convert frame start times to time from injection in seconds.
    """
    injection_time_seconds = injection_time.hour*3600 + injection_time.minute*60 + injection_time.second
    start_time_seconds = [
        (start_time.hour*3600
            + start_time.minute*60
            + start_time.second)
            - injection_time_seconds for start_time in start_times]
    return start_time_seconds


def single_exp(t, a, r):
    return a*np.exp(-r*t)


def two_exp(t, a0, r0, a1, r1):
    return a0*np.exp(-r0*t)+a1*np.exp(-r1*t)


def decay_correct(
        frame_times,
        activity_uncorrected,
        half_life: float
        ):
    decay_constant = np.log(2)/half_life
    frame_decay_factor = np.exp(decay_constant*frame_times)
    activity_corrected = frame_decay_factor * activity_uncorrected
    return activity_corrected

def exp_fit_dosimetry(
        frame_ref_time: list[float],
        activity_uncorrected: list[float],
        half_life: float,
        ):
    activity_corrected = decay_correct(
        frame_times=frame_ref_time,
        activity_uncorrected=activity_uncorrected,
        half_life=half_life
    )
    try:
        popt, pcov = curve_fit(
            f=two_exp,
            xdata=frame_ref_time,
            ydata=activity_corrected,
            p0=[1e-1,1e-4,-1e-1,1e-4],
            method='lm',
            maxfev=10000
        )
    except:
        popt = [1,1,1,1]
        pcov = [1,1,1,1]
    return popt, pcov


def time_integrated_activity(
        exp_coeffs: list[float],
        half_life: float
        ):
    decay_constant = np.log(2)/half_life
    return exp_coeffs[0]/(exp_coeffs[1]+decay_constant) + exp_coeffs[2]/(exp_coeffs[3]+decay_constant)


def run_dosimetry(
        roi_data: pd.DataFrame,
        plots_save: str
        ):
    """
    Run dosimetry processing to produce time-integrated activity, TAC plots, and save into an excel spreadsheet.

    roi_data: must have 'Scan' as the index
    """
    tia = []
    scans = roi_data.index.unique()
    plotting_df = pd.DataFrame([{}])
    tia_df = pd.DataFrame([{}])
    for scan in scans:
        scan_roi_data = roi_data.filter(like=scan,axis=0)
        organ_list = scan_roi_data['Organ'].unique()
        for organ in organ_list:
            temp_df = scan_roi_data.set_index('Organ')
            organ_roi_data = temp_df.filter(like=organ,axis=0)
            organ_frame_start_times = convert_frame_time_seconds(
                start_times=organ_roi_data['Scan Time'],
                injection_time=organ_roi_data['Injection Time'].iloc[0]
                )
            organ_frame_midpoint_times = time_midpoint(
                start_times=organ_frame_start_times,
                frame_durations=organ_roi_data['Frame Duration (s)'],
                half_life=organ_roi_data['Half Life (s)'].iloc[0]
                )
            organ_dose_ratio = organ_dose(
                activity_cps=organ_roi_data['Activity (cps)'],
                organ_volume=organ_roi_data['Organ Volume (mL)'],
                calibration_factor=organ_roi_data['Calibration Factor'].iloc[0],
                injected_dose=organ_roi_data['Injected Dose (mCi)'].iloc[0],
                scale_factor=organ_roi_data['Scale Factor'].iloc[0]
                )
            organ_dose_ratio_corrected = decay_correct(
                frame_times=organ_frame_midpoint_times,
                activity_uncorrected=organ_dose_ratio,
                half_life=organ_roi_data['Half Life (s)'].iloc[0]
            )
            popt, pcov = exp_fit_dosimetry(
                frame_ref_time=organ_frame_midpoint_times,
                activity_uncorrected=organ_dose_ratio,
                half_life=organ_roi_data['Half Life (s)'].iloc[0]
                )
            tia = time_integrated_activity(
                exp_coeffs=popt,
                half_life=organ_roi_data['Half Life (s)'].iloc[0]
            )/3600
            tia_event = pd.DataFrame([{'scan': scan, 'organ': organ, 'time_integrated_activity_hr': tia}])
            tia_df = pd.concat([tia_df,tia_event],ignore_index=True)
            for i,frame_ref_time in enumerate(organ_frame_midpoint_times):
                organ_event = pd.DataFrame([{
                    'scan': scan,
                    'organ': organ,
                    'frame_ref_time': frame_ref_time,
                    'dose_ratio': organ_dose_ratio[i],
                    'dose_ratio_corrected': organ_dose_ratio_corrected[i],
                    'dose_ratio_fit': two_exp(organ_frame_midpoint_times[i],*popt)
                    }])
                plotting_df = pd.concat([plotting_df,organ_event],ignore_index=True)
                plotting_df = pd.concat([plotting_df,organ_event],ignore_index=True)
    plotting_df.to_csv(f'{plots_save}/dosimetry_tacs.csv')
    dose_plot = sns.relplot(x='frame_ref_time',y='dose_ratio',data=plotting_df,col='organ',hue='scan',facet_kws=dict(sharey=False))
    dose_plot.figure.savefig(f'{plots_save}/dosimetry_dose_ratio.png')
    g = sns.relplot(x='frame_ref_time',y='dose_ratio_corrected',data=plotting_df,col='organ',hue='scan',facet_kws=dict(sharey=False))
    for organ, ax in g.axes_dict.items():
        temp_df = plotting_df.set_index('organ')
        temp_df = temp_df.filter(like=organ,axis=0)
        sns.lineplot(x='frame_ref_time',y='dose_ratio_fit',data=temp_df,hue='scan',ax=ax)
    tia_df.to_csv(f'{plots_save}/time_integrated_activity.csv')
