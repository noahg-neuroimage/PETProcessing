"""
Command-line interface (CLI) for Partial Volume Correction (PVC) using SGTM and PETPVC methods.
This module provides a CLI to apply PVC to PET images using either the SGTM method or the PETPVC package.
It uses argparse to handle command-line arguments and chooses the appropriate method based on the provided input.
The user must provide:
    * PET image file path
    * Segmentation image file path
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

from ..preproc.partial_volume_corrections import PetPvc
from ..preproc.symmetric_geometric_transfer_matrix import Sgtm


def sanitize_path(path: str) -> str:
    """
    Sanitize the given path for Docker command usage by converting it to POSIX format.
    Args:
        path (str): The file path to sanitize.
    Returns:
        str: The sanitized POSIX path.
    """
    return path.replace(os.sep, '/')


def petpvc_cli_run(args):
    """
    Apply the PETPVC method for Partial Volume Correction.
    Args:
        pet_img (nib.Nifti1Image): The 3D PET image.
        roi_img (nib.Nifti1Image): The 3D Segmentation image.
        fwhm (float or Tuple): Full Width at Half Maximum for Gaussian blurring.
        output_path (str): The path where the output image will be saved.
        debug (bool, optional): Enable debug mode for detailed logs.
        verbose (bool, optional): Print additional information.
    """
    if not args.output_path:
        raise ValueError("The --output-path argument is required for the PETPVC method.")

    input_image = sanitize_path(args.input_image)
    output_path = sanitize_path(args.output_path)
    segmentation_image = sanitize_path(args.segmentation_image) if args.segmentation_image else None

    if args.verbose:
        print(f"PET Path: {input_image}")
        print(f"Segmentation Path: {args.segmentation_image}")
        print(f"Output Path: {output_path}")
        if segmentation_image:
            print(f"Mask Path: {segmentation_image}")
        print(f"FWHM: {args.fwhm}")
        print(f"Method: {args.method}")

    petpvc_handler = PetPvc()

    try:
        petpvc_handler.run_petpvc(
            pet_4d_filepath=input_image,
            output_filepath=output_path,
            pvc_method=args.method,
            psf_dimensions=args.fwhm,
            mask_filepath=segmentation_image,
            verbose=args.verbose,
            debug=args.debug
        )
    except docker.errors.ContainerError as e:
        print(f"ContainerError: {e}")
        print("Command failed inside the Docker container. Check the above error message for details.")


def sgtm_cli_run(args):
    """
    Apply the SGTM method for Partial Volume Correction.
    Args:
        pet_img (nib.Nifti1Image): The 3D PET image.
        roi_img (nib.Nifti1Image): The 3D Segmentation image.
        fwhm (float or Tuple): Full Width at Half Maximum for Gaussian blurring.
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Labels, corrected PET values, and condition number of the omega matrix.
    """
    sgtm_obj = Sgtm(input_image_path=args.input_image,
                    segmentation_image_path=args.segmentation_image,
                    fwhm=args.fwhm,
                    out_tsv_path=args.output_path)
    if args.verbose:
        print("Labels:", sgtm_obj.sgtm_result[0])
        print("Corrected values:", sgtm_obj.sgtm_result[1])
        print("Condition number:", sgtm_obj.sgtm_result[2])


def main():
    """
    Main function to handle command-line arguments and apply the appropriate PVC method.
    """
    parser = argparse.ArgumentParser(
        prog="PVC CLI",
        description="Apply Partial Volume Correction (PVC) to PET images using SGTM or PETPVC methods.",
        epilog="Example of usage: pet-cli-pvc --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 8.0"
    )

    parser.add_argument("-m","--method", required=True, help="PVC method to use (SGTM or PETPVC method).")
    parser.add_argument("-i","--input-image", required=True, help="Path to the PET image file.")
    parser.add_argument("-s","--segmentation_image", required=True,
                        help="Path to the Segmentation image file.")
    parser.add_argument("-f","--fwhm", required=True, type=float,
                        help="Full Width at Half Maximum for Gaussian blurring (Tuple or single "
                             "float) in mm.")
    parser.add_argument("-o","--output-path", help="Path to the output image file (for PETPVC method).")
    parser.add_argument("-v","--verbose", action="store_true", help="Print additional information.")
    parser.add_argument("-d","--debug", action="store_true", help="Enable debug mode (for PETPVC method).")

    args = parser.parse_args()

    if args.method.lower() == "sgtm":
        sgtm_cli_run(args=args)
    else:
        petpvc_cli_run(args=args)

if __name__ == "__main__":
    main()
