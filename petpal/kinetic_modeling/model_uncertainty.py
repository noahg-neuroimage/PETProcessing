"""Tools for calculating uncertainty for application to kinetic models."""
import numpy as np

from ..utils.time_activity_curve import TimeActivityCurve


class ModelUncertainty:
    """Set uncertainty for kinetic modeling. Includes options for constant,
    calculated, or preset uncertainty.    
    """
    def __init__(self,
                 time_activity_curve: TimeActivityCurve):
        """Initialize ModelUncertainty with provided arguments.

        Args:
            time_activity_curve (TimeActivityCurve): The time activity curve used to determine
                uncertainties to pass onto the kinetic model.
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


    def __call__(self, uncertainty_method: str) -> np.ndarray:
        """Get the model uncertainty corresponding to the identified method.
        
        Args:
            uncertainty_methood (str): model uncertainty type to apply to the model.

        Returns:
            uncertainty (np.ndarray): The uncertainty applied to each time frame in the model.

        Raises:
            ValueError: If `uncertainty_method` is not one of: 'constant', 'calculated', or 
                'provided'.
        """
        match uncertainty_method:
            case 'constant':
                return self.constant_uncertainty
            case 'provided':
                return self.provided_uncertainty
            case 'calculated':
                return self.calculated_uncertainty
            case _:
                raise ValueError("uncertainty_method must be one of: 'constant','calculated', "
                                f"'provided'. Got {uncertainty_method}.")
