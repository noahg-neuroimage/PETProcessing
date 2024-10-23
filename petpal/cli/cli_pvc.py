"""
Command-line interface (CLI) for Partial Volume Correction (PVC) using SGTM and PETPVC methods.
This module provides a CLI to apply PVC to PET images using either the SGTM method or the PETPVC package.
It uses argparse to handle command-line arguments and chooses the appropriate method based on the provided input.
The user must provide:
    * PET image file path
    * ROI image file path
    * FWHM for Gaussian blurring
    * PVC method ('SGTM' or any other method for PETPVC)
    * Additional options for PETPVC
Example usage:
    Using SGTM method:
        .. code-block:: bash
            pvc_cli.py --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm (8.0, 7.0, 7.0)
    Using PETPVC method:
        .. code-block:: bash
            pvc_cli.py --method RBV --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 6.0 --output-path /path/to/output_image.nii
See Also:
    SGTM and PETPVC methods implementation modules.
    :mod:`SGTM <pet_cli.symmetric_geometric_transfer_matrix>` - module for performing symmetric Geometric Transfer Matrix PVC.
    :mod:`PETPVC <pet_cli.partial_volume_corrections>` - wrapper for PETPVC package.
"""

import os
import argparse

import docker
import docker.errors
import nibabel as nib

from ..preproc.partial_volume_corrections import PetPvc
from ..preproc.symmetric_geometric_transfer_matrix import sgtm


def sanitize_path(path: str) -> str:
    """
    Sanitize the given path for Docker command usage by converting it to POSIX format.
    Args:
        path (str): The file path to sanitize.
    Returns:
        str: The sanitized POSIX path.
    """
    return path.replace(os.sep, '/')


def main():
    """
    Main function to handle command-line arguments and apply the appropriate PVC method.
    """
    parser = argparse.ArgumentParser(
        prog="PVC CLI",
        description="Apply Partial Volume Correction (PVC) to PET images using SGTM or PETPVC methods.",
        epilog="Example of usage: pet-cli-pvc --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 8.0"
    )

    parser.add_argument("--method", required=True, help="PVC method to use (SGTM or PETPVC method).")
    parser.add_argument("--pet-path", required=True, help="Path to the PET image file.")
    parser.add_argument("--roi-path", required=True, help="Path to the ROI image file.")
    parser.add_argument("--fwhm", required=True, type=float,
                        help="Full Width at Half Maximum for Gaussian blurring (Tuple or single float).")
    parser.add_argument("--output-path", help="Path to the output image file (for PETPVC method).")
    parser.add_argument("--verbose", action="store_true", help="Print additional information.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (for PETPVC method).")

    args = parser.parse_args()

    pet_img = nib.load(args.pet_path)
    roi_img = nib.load(args.roi_path)

    if args.method.lower() == "sgtm":
        """
        Apply the SGTM method for Partial Volume Correction.
        Args:
            pet_img (nib.Nifti1Image): The 3D PET image.
            roi_img (nib.Nifti1Image): The 3D ROI image.
            fwhm (float or Tuple): Full Width at Half Maximum for Gaussian blurring.
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Labels, corrected PET values, and condition number of the omega matrix.
        """
        labels, corrected_values, cond_number = sgtm(pet_img, roi_img, args.fwhm)
        if args.verbose:
            print("Labels:", labels)
            print("Corrected values:", corrected_values)
            print("Condition number:", cond_number)
    else:
        """
        Apply the PETPVC method for Partial Volume Correction.
        Args:
            pet_img (nib.Nifti1Image): The 3D PET image.
            roi_img (nib.Nifti1Image): The 3D ROI image.
            fwhm (float or Tuple): Full Width at Half Maximum for Gaussian blurring.
            output_path (str): The path where the output image will be saved.
            debug (bool, optional): Enable debug mode for detailed logs.
            verbose (bool, optional): Print additional information.
        """
        if not args.output_path:
            raise ValueError("The --output-path argument is required for the PETPVC method.")

        pet_path = sanitize_path(args.pet_path)
        output_path = sanitize_path(args.output_path)
        roi_path = sanitize_path(args.roi_path) if args.roi_path else None

        if args.verbose:
            print(f"PET Path: {pet_path}")
            print(f"ROI Path: {args.roi_path}")
            print(f"Output Path: {output_path}")
            if roi_path:
                print(f"Mask Path: {roi_path}")
            print(f"FWHM: {args.fwhm}")
            print(f"Method: {args.method}")

        petpvc_handler = PetPvc()

        try:
            petpvc_handler.run_petpvc(
                pet_4d_filepath=pet_path,
                output_filepath=output_path,
                pvc_method=args.method,
                psf_dimensions=args.fwhm,
                mask_filepath=roi_path,
                verbose=args.verbose,
                debug=args.debug
            )
        except docker.errors.ContainerError as e:
            print(f"ContainerError: {e}")
            print("Command failed inside the Docker container. Check the above error message for details.")


if __name__ == "__main__":
    main()
