"""Tools for calculating uncertainty for application to kinetic models."""
import numpy as np

from ..utils.time_activity_curve import TimeActivityCurve


class ModelUncertainty:
    """Determine weighting scheme for Time Activity Curves. Includes options for constant,
    calculated, or preset weighting.    
    """
    def __init__(self,
                 time_activity_curve: TimeActivityCurve):
        """Initialize ModelUncertainty with provided arguments.

        Args:
            time_activity_curve (TimeActivityCurve): The time activity curve on which uncertainty
                are applied.
        """
        self.time_activity_curve = time_activity_curve


    @property
    def constant_uncertainty(self):
        """Get constant uncertainty for the model.
        """
        uncertainty = np.ones_like(self.time_activity_curve.activity)
        return uncertainty


    @property
    def provided_uncertainty(self):
        """Get user provided uncertainty for the model.
        """
        return self.time_activity_curve.uncertainty


    @property
    def calculated_uncertainty(self):
        """The calculated uncertainty for the model.

        Currently placeholder function.
        """
        return None


    def __call__(self, weight_method: str) -> np.ndarray:
        """Get the model uncertainty corresponding to the identified method.
        
        Args:
            weight_methood (str): model weight type to apply to the model.

        Returns:
            uncertainty (np.ndarray): The weight applied to each time frame in the model.

        Raises:
            ValueError: If `weight_method` is not one of: 'constant', 'calculated', or 'provided'.
        """
        match weight_method:
            case 'constant':
                return self.constant_uncertainty
            case 'provided':
                return self.provided_uncertainty
            case 'calculated':
                return self.calculated_uncertainty
            case _:
                raise ValueError("weight_method must be one of: 'constant','calculated', "
                                f"'provided'. Got {weight_method}.")
