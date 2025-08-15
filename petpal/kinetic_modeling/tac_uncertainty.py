"""Tools for calculating TAC uncertainty for application to kinetic models."""
import numpy as np

from ..utils.time_activity_curve import TimeActivityCurve


class TacUncertainty:
    """Set uncertainty for kinetic modeling. Includes options for constant,
    calculated, or preset uncertainty.    
    """
    def __init__(self,
                 time_activity_curve: TimeActivityCurve):
        """Initialize TacUncertainty with provided arguments.

        Args:
            time_activity_curve (TimeActivityCurve): The time activity curve used to determine
                uncertainties to pass onto the kinetic model.
        """
        self.time_activity_curve = time_activity_curve


    @property
    def constant_uncertainty(self) -> np.ndarray:
        """Get constant uncertainty for the TAC.
        """
        uncertainty = np.ones_like(self.time_activity_curve.activity)
        return uncertainty


    @property
    def tac_uncertainty(self) -> np.ndarray:
        """Get uncertainty stored in the TAC itself to pass on to the kinetic model.

        Raises:
            ValueError: If the supplied TAC uncertainty has any NaN values.
        
        Returns:
            uncertainty_from_tac (np.ndarray): Uncertainty stored in the supplied TAC.
        """
        uncertainty_from_tac = self.time_activity_curve.uncertainty
        nan_locations = np.isnan(uncertainty_from_tac)
        if np.any(nan_locations):
            raise ValueError("Supplied TAC has NaN values in stored uncertianty array. Use an "
                             "uncertainty method other than 'tac'.")
        return uncertainty_from_tac


    @property
    def calculated_uncertainty(self):
        """The calculated uncertainty for the TAC.

        Currently placeholder function.
        """
        raise NotImplementedError('Calculated uncertainty model for PET not yet implemented. Use '
                                  'constant uncertainty instead.')


    def __call__(self, uncertainty_method: str='constant') -> np.ndarray:
        """Get the tac uncertainty corresponding to the identified method.
        
        Args:
            uncertainty_method (str): Uncertainty type to apply to the tac. Default
                'constant'.

        Returns:
            uncertainty (np.ndarray): The uncertainty applied to each time frame in the TAC.

        Raises:
            ValueError: If `uncertainty_method` is not one of: 'constant', 'calculated', or 
                'tac'.
        """
        match uncertainty_method.lower():
            case 'constant':
                return self.constant_uncertainty
            case 'tac':
                return self.tac_uncertainty
            case 'calculated':
                return self.calculated_uncertainty
            case _:
                raise ValueError("uncertainty_method must be one of: 'constant','calculated', "
                                f"'tac'. Got {uncertainty_method}.")
