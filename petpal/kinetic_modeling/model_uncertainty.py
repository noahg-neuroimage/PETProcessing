"""Tools for calculating weights for application to kinetic models."""
import numpy as np
import ants

from ..utils.time_activity_curve import TimeActivityCurve
from ..utils.image_io import get_half_life_from_nifti
from ..utils.scan_timing import ScanTimingInfo


class ModelUncertainty:
    """Determine weighting scheme for Time Activity Curves. Includes options for constant,
    calculated, or preset weighting.    
    """
    def __init__(self,
                 time_activity_curve: TimeActivityCurve,
                 input_image_path: str=None):
        """Initialize TacWeight with provided arguments.

        Args:
            time_activity_curve (TimeActivityCurve): The time activity curve on which weights are
                applied.
            input_image_path (str): Path to the PET image used to create the TAC supplied to
                object. Used only to retrieve scan timing and half life information for calculated
                TAC weights. Default None.
        """
        self.time_activity_curve = time_activity_curve


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
        return self.time_activity_curve.uncertainty


    @property
    def calculated_weights(self):
        """The calculated weights for the TAC.
        """
        self.validate_scan_timing()
        weights = self.weight_tac_decay()
        return weights


    def __call__(self, weight_method: str) -> np.ndarray:
        """Get the TAC weights corresponding to the identified method.
        
        Args:
            weight_methood (str): TAC weight type to apply to the model.

        Returns:
            weights (np.ndarray): The weight applied to each time frame in the model.

        Raises:
            ValueError: If `weight_method` is not one of: 'constant', 'calculated', or 'provided'.
        """
        match weight_method:
            case 'constant':
                return self.constant_weights
            case 'provided':
                return self.provided_weights
            case 'calculated':
                return self.calculated_weights
            case _:
                raise ValueError("weight_method must be one of: 'constant','calculated', "
                                f"'provided'. Got {weight_method}.")
