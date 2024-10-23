"""
This module contains the FitTacWithRTMs class, used to fit kinetic models to a target and
reference Time Activity Curve.
"""
from typing import Union
import numpy as np
from petpal.kinetic_modeling.reference_tissue_models import (fit_frtm2_to_tac,
                                                             fit_frtm2_to_tac_with_bounds,
                                                             fit_frtm_to_tac,
                                                             fit_frtm_to_tac_with_bounds,
                                                             fit_mrtm2_2003_to_tac,
                                                             fit_mrtm_2003_to_tac,
                                                             fit_mrtm_original_to_tac,
                                                             fit_srtm2_to_tac,
                                                             fit_srtm2_to_tac_with_bounds,
                                                             fit_srtm_to_tac,
                                                             fit_srtm_to_tac_with_bounds)


class FitTACWithRTMs:
    r"""
    A class used to fit a kinetic model to both a target and a reference Time Activity Curve (TAC).

    The :class:`FitTACWithRTMs` class simplifies the process of kinetic model fitting by providing
    methods for validating input data, choosing a model to fit, and then performing the fit. It
    takes in raw intensity values of TAC for both target and reference regions as inputs, which are
    then used in curve fitting.

    This class supports various kinetic models, including but not limited to: the simplified and
    full reference tissue models (SRTM & FRTM), and the multilinear reference tissue models
    (Orignial MRMT, MRTM & MRTM2). Each model type can be bounded or unbounded.

    The fitting result contains the estimated kinetic parameters depending on the chosen model.

    Attributes:
        tac_times_in_minutes (np.ndarray): The array representing the time-points for both TACs.
        target_tac_vals (np.ndarray): The target TAC values.
        reference_tac_vals (np.ndarray): The reference TAC values.
        method (str): Optional. The kinetic model to use. Defaults to 'mrtm'.
        bounds (np.ndarray): Optional. Parameter bounds for the specified kinetic model. Defaults
            to None.
        t_thresh_in_mins (float): Optional. The times at which the reference TAC was sampled. 
            Defaults to None.
        k2_prime (float): Optional. The estimated efflux rate constant for the non-displaceable 
            compartment. Defaults to None.
        fit_results (np.ndarray): The result of the fit.

    Example:
        The following example shows how to use the :class:`FitTACWithRTMs` class to fit the SRTM to
        a target and reference TAC.

        .. code-block:: python

            import numpy as np
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.reference_tissue_models as pet_rtms

            # loading the input tac to generate a reference region tac
            input_tac_times, input_tac_vals = np.asarray(np.loadtxt("../../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt").T,
                                                         float)

            # generating a reference region tac
            ref_tac_times, ref_tac_vals = pet_tcm.generate_tac_1tcm_c1_from_tac(tac_times_in_minutes=input_tac_times, tac_vals=input_tac_vals,
                                                                                k1=1.0, k2=0.2)

            # generating an SRTM tac
            srtm_tac_vals = pet_rtms.calc_srtm_tac(tac_times_in_minutes=ref_tac_times, ref_tac_vals=ref_tac_vals, r1=1.0, k2=0.25, bp=3.0)

            rtm_analysis = pet_rtms.FitTACWithRTMs(target_tac_vals=srtm_tac_vals,
                                                tac_times_in_minutes=ref_tac_times,
                                                reference_tac_vals=ref_tac_vals,
                                                method='srtm')

            # performing the fit
            rtm_analysis.fit_tac_to_model()
            fit_results = rtm_analysis.fit_results[1]


    This will give you the kinetic parameter values of the SRTM for the provided TACs.

    See Also:
        * :meth:`validate_bounds`
        * :meth:`validate_method_inputs`
        * :meth:`fit_tac_to_model`

    """
    def __init__(self,
                 tac_times_in_minutes: np.ndarray,
                 target_tac_vals: np.ndarray,
                 reference_tac_vals: np.ndarray,
                 method: str = 'mrtm',
                 bounds: Union[None, np.ndarray] = None,
                 t_thresh_in_mins: float = None,
                 k2_prime: float = None):
        r"""
        Initialize the FitTACWithRTMs object with specified parameters.

        This method sets up input parameters and validates them. We check if the bounds are correct
        for the given 'method', and we make sure that any fitting threshold are defined for the
        MRTM analyses.


        Args:
            tac_times_in_minutes (np.ndarray): The array representing the time-points for both TACs.
            target_tac_vals (np.ndarray): The array representing the target TAC values.
            reference_tac_vals (np.ndarray): The array representing values of the reference TAC.
            method (str, optional): The kinetics method to be used. Default is 'mrtm'.
            bounds (Union[None, np.ndarray], optional): Bounds for kinetic parameters used in
                optimization. None represents absence of bounds. Default is None.
            t_thresh_in_mins (float, optional): Threshold for time separation in minutes. Default
                is None.
            k2_prime (float, optional): The estimated rate constant related to the flush-out rate
                of the reference compartment. Default is None.

        Raises:
            ValueError: If a parameter necessary for chosen method is not provided.
            AssertionError: If rate constant k2_prime is non-positive.
        """

        self.tac_times_in_minutes: np.ndarray = tac_times_in_minutes
        self.target_tac_vals: np.ndarray = target_tac_vals
        self.reference_tac_vals: np.ndarray = reference_tac_vals
        self.method: str = method.lower()
        self.bounds: Union[None, np.ndarray] = bounds
        self.validate_bounds()

        self.t_thresh_in_mins: float = t_thresh_in_mins
        self.k2_prime: float = k2_prime

        self.validate_method_inputs()

        self.fit_results: Union[None, np.ndarray] = None

    def validate_method_inputs(self):
        r"""Validates the inputs for different methods

        This method validates the inputs depending on the chosen method in the object.

        - If the method is of type 'mrtm', it checks if `t_thresh_in_mins` is defined and positive.
        - If the method ends with a '2' (the reduced/modified methods), it checks if `k2_prime` is 
            defined and positive.

        Raises:
            ValueError: If ``t_thresh_in_mins`` is not defined while the method starts with 'mrtm'.
            AssertionError: If ``t_thresh_in_mins`` is not a positive number.
            ValueError: If ``k2_prime`` is not defined while the method ends with '2'.
            AssertionError: If ``k2_prime`` is not a positive number.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.method.startswith("mrtm"):
            if self.t_thresh_in_mins is None:
                raise ValueError(
                    "t_t_thresh_in_mins must be defined if method is 'mrtm'")
            else:
                assert self.t_thresh_in_mins >= 0, "t_thresh_in_mins must be a positive number."
        if self.method.endswith("2"):
            if self.k2_prime is None:
                raise ValueError("k2_prime must be defined if we are using the reduced models: "
                                 "FRTM2, SRTM2, and MRTM2.")
            assert self.k2_prime >= 0, "k2_prime must be a positive number."

    def validate_bounds(self):
        r"""Validates the bounds for different methods

        This method validates the shape of the bounds depending on the chosen method in the object.

        - If the method is 'srtm', it checks that bounds shape is (3, 3).
        - If the method is 'frtm', it checks that bounds shape is (4, 3).

        Raises:
            AssertionError: If the bounds shape for method 'srtm' is not (3, 3)
            AssertionError: If the bounds shape for method 'frtm' is not (4, 3).
            ValueError: If the method is not 'srtm' or 'frtm' while providing bounds.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.bounds is not None:
            num_params, num_vals = self.bounds.shape
            if self.method == "srtm":
                assert num_params == 3 and num_vals == 3, ("The bounds have the wrong shape. "
                                                           "Bounds must be (start, lo, hi) for each"
                                                           "of the fitting "
                                                           "parameters: r1, k2, bp")
            elif self.method == "frtm":
                assert num_params == 4 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k2, k3, k4")

            elif self.method == "srtm2":
                assert num_params == 2 and num_vals == 3, ("The bounds have the wrong shape. Bounds"
                                                           "must be (start, lo, hi) "
                                                           "for each of the"
                                                           " fitting parameters: r1, bp")
            elif self.method == "frtm2":
                assert num_params == 3 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k3, k4")
            else:
                raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'srtm2' or "
                                 "'frtm2' if bounds are "
                                 f"provided. Got {self.method}.")

    def fit_tac_to_model(self):
        r"""Fits TAC vals to model

        This method fits the target TAC values to the model depending on the chosen method in the
        object.

        - If the method is 'srtm' or 'frtm', and bounds are provided, fitting functions with bounds
            are used.
        - If the method is 'srtm' or 'frtm', and bounds are not provided, fitting functions without
            bounds are used.
        - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.

        Raises:
            ValueError: If the method name is invalid and not one of 'srtm', 'frtm',
                'mrtm-original', 'mrtm' or 'mrtm2'.


        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_srtm2_to_tac_with_bounds`
            * :func:`fit_srtm2_to_tac`
            * :func:`fit_frtm2_to_tac_with_bounds`
            * :func:`fit_frtm2_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.method == "srtm":
            if self.bounds is not None:
                self.fit_results = fit_srtm_to_tac_with_bounds(tac_times_in_minutes=self.tac_times_in_minutes,
                                                               tgt_tac_vals=self.target_tac_vals,
                                                               ref_tac_vals=self.reference_tac_vals,
                                                               r1_bounds=self.bounds[0], k2_bounds=self.bounds[1],
                                                               bp_bounds=self.bounds[2])
            else:
                self.fit_results = fit_srtm_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                   tgt_tac_vals=self.target_tac_vals,
                                                   ref_tac_vals=self.reference_tac_vals)

        elif self.method == "srtm2":
            if self.bounds is not None:
                self.fit_results = fit_srtm2_to_tac_with_bounds(tac_times_in_minutes=self.tac_times_in_minutes,
                                                                tgt_tac_vals=self.target_tac_vals,
                                                                ref_tac_vals=self.reference_tac_vals,
                                                                k2_prime=self.k2_prime, r1_bounds=self.bounds[0],
                                                                bp_bounds=self.bounds[1])
            else:
                self.fit_results = fit_srtm2_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                    tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_vals=self.reference_tac_vals, k2_prime=self.k2_prime)
        elif self.method == "frtm":
            if self.bounds is not None:
                self.fit_results = fit_frtm_to_tac_with_bounds(tac_times_in_minutes=self.tac_times_in_minutes,
                                                               tgt_tac_vals=self.target_tac_vals,
                                                               ref_tac_vals=self.reference_tac_vals,
                                                               r1_bounds=self.bounds[0], k2_bounds=self.bounds[1],
                                                               k3_bounds=self.bounds[2], k4_bounds=self.bounds[3])
            else:
                self.fit_results = fit_frtm_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                   tgt_tac_vals=self.target_tac_vals,
                                                   ref_tac_vals=self.reference_tac_vals)

        elif self.method == "frtm2":
            if self.bounds is not None:
                self.fit_results = fit_frtm2_to_tac_with_bounds(tac_times_in_minutes=self.tac_times_in_minutes,
                                                                tgt_tac_vals=self.target_tac_vals,
                                                                ref_tac_vals=self.reference_tac_vals,
                                                                k2_prime=self.k2_prime, r1_bounds=self.bounds[0],
                                                                k3_bounds=self.bounds[1], k4_bounds=self.bounds[2])
            else:
                self.fit_results = fit_frtm2_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                    tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_vals=self.reference_tac_vals, k2_prime=self.k2_prime)

        elif self.method == "mrtm-original":
            self.fit_results = fit_mrtm_original_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                        tgt_tac_vals=self.target_tac_vals,
                                                        ref_tac_vals=self.reference_tac_vals,
                                                        t_thresh_in_mins=self.t_thresh_in_mins)

        elif self.method == "mrtm":
            self.fit_results = fit_mrtm_2003_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                    tgt_tac_vals=self.target_tac_vals,
                                                    ref_tac_vals=self.reference_tac_vals,
                                                    t_thresh_in_mins=self.t_thresh_in_mins)

        elif self.method == "mrtm2":
            self.fit_results = fit_mrtm2_2003_to_tac(tac_times_in_minutes=self.tac_times_in_minutes,
                                                     tgt_tac_vals=self.target_tac_vals,
                                                     ref_tac_vals=self.reference_tac_vals,
                                                     t_thresh_in_mins=self.t_thresh_in_mins, k2_prime=self.k2_prime)
        else:
            raise ValueError("Invalid method! Must be either 'srtm', 'frtm', 'mrtm-original', "
                             f"'mrtm' or 'mrtm2'. Got {self.method}.")
