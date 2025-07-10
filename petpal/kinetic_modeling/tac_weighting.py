"""Tools for calculating weights for application to kinetic models."""
import numpy as np

from ..utils.time_activity_curve import TimeActivityCurve
from ..utils.image_io import get_half_life_from_nifti
from ..utils.scan_timing import ScanTimingInfo

class TacWeight:
    """Determine weighting scheme for Time Activity Curves. Includes options for constant,
    calculated, or preset weighting.    
    """
    def __init__(self,
                 weight_method: str,
                 time_activity_curve: TimeActivityCurve,
                 input_image_path: str=None):
        """Initialize TacWeight with provided arguments.

        Args:
            weight_method (str): Weighting method to apply during kinetic modeling.
            time_activity_curve (TimeActivityCurve): The time activity curve on which weights are
                applied.
        """
        self.weight_method = weight_method
        self.time_activity_curve = time_activity_curve
        self.input_image_path = input_image_path
        self.weights = None

        self.validate_weight_method()

    def validate_weight_method(self):
        """Validate the weight_method input parameter is one of: constant, calculated, or provided.
        """
        if self.weight_method not in ['constant','calculated','provided']:
            raise ValueError("weight_method must be one of: 'constant','calculated','provided'."
                            f"Got {self.weight_method}.")


    def weight_tac_simple(self,
                          tac_durations_in_minutes: np.ndarray,
                          tac_vals: np.ndarray) -> np.ndarray:
        """Weight a Time Activity Curve (TAC) based on variance. This function applies the simple
        frame time length to activity ratio found in
        http://www.turkupetcentre.net/petanalysis/model_weighting.html.

        Args:
            tac_durations_in_minutes (np.ndarray): Duration of each frame in the TAC in minutes.
            tac_vals (np.ndarray): Activity values for each frame in the TAC.

        Returns:
            tac_weights (np.ndarray): Weights to be applied to the TAC during fitting process.
        """
        tac_weights = tac_durations_in_minutes/tac_vals
        tac_vals_where_zero = np.where(tac_vals==0)
        tac_weights[tac_vals_where_zero] = 0
        return tac_weights


    def weight_tac_decay(self,
                         tac_durations_in_minutes: np.ndarray,
                         tac_vals: np.ndarray,
                         tac_times_in_minutes: np.ndarray,
                         half_life_in_minutes: np.ndarray) -> np.ndarray:
        """Weight a Time Activity Curve (TAC) based on variance. This function applies the simple
        frame time length to activity ratio found in
        http://www.turkupetcentre.net/petanalysis/model_weighting.html with an extra factor for
        decay correction.

        Args:
            tac_durations_in_minutes (np.ndarray): Duration of each frame in the TAC in minutes.
            tac_vals (np.ndarray): Activity values for each frame in the TAC.
            tac_times_in_minutes (np.ndarray): Frame times for the TAC in minutes.
            half_life_in_minutes (np.ndarray): Half life of the radioisotope in minutes.

        Returns:
            tac_weights (np.ndarray): Weights to be applied to the TAC during fitting process.
        """
        decay_factor = np.exp(-np.log(2) / half_life_in_minutes * tac_times_in_minutes)
        tac_weights = tac_durations_in_minutes * decay_factor / tac_vals
        tac_vals_where_zero = np.where(tac_vals==0)
        tac_weights[tac_vals_where_zero] = 0
        return tac_weights


    def convert_weights_to_sigma(self, tac_weights: np.ndarray) -> np.ndarray:
        r"""Convert TAC weights to sigma (standard deviation) values. Calculated as
        :math:`\sigma=w^{-1/2}`. Returns zero as the sigma value if the weight at that time point is
        zero.

        Args:
            tac_weights (np.ndarray): Weights calculated using :meth:`weight_tac_simple` or
            `weight_tac_decay`.

        Returns:
            tac_sigma (np.ndarray): Array of sigmas calculated from the weights.
        """
        tac_sigma = np.power(tac_weights,-1/2)
        tac_weights_where_zero = np.where(tac_weights==0)
        tac_sigma[tac_weights_where_zero] = np.inf
        return tac_sigma


    @property
    def constant_weights(self):
        """Get constant weights for the TAC.
        """
        weights = np.ones_like(self.time_activity_curve.activity)
        return weights


    @property
    def provided_weights(self):
        """Get user provided weights for the TAC.
        """
        weights = self.time_activity_curve.uncertainty
        return weights

    @property
    def calculated_weights(self):
        """Get calculated weights for the TAC.

        weights = self.weight_tac_decay()
        """
        half_life = get_half_life_from_nifti(image_path=self.input_image_path)
        scan_timing = ScanTimingInfo.from_nifti(image_path=self.input_image_path)
        weights = self.weight_tac_decay(tac_durations_in_minutes=scan_timing.duration_in_mins,
                                        tac_vals=self.time_activity_curve.activity,
                                        tac_times_in_minutes=scan_timing.center_in_mins,
                                        half_life_in_minutes=half_life)
        return weights
