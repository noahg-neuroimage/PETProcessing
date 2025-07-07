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
import argparse

from ..preproc.symmetric_geometric_transfer_matrix import Sgtm


def sgtm_cli_run(args):
    """
    Apply the SGTM method for Partial Volume Correction.
    """
    sgtm_obj = Sgtm(input_image_path=args.input_image,
                    segmentation_image_path=args.segmentation_image,
                    fwhm=args.fwhm)
    sgtm_obj(output_path=args.output_path)

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

if __name__ == "__main__":
    main()
