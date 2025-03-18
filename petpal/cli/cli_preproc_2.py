"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * The sub-command. Options: 'weighted-sum', 'motion-correct', 'register-pet', or 'write-tacs'.
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
      such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
      preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    

Examples:
(Need to fix these)

See Also:
    * :mod:`petpal.preproc.image_operations_4d` - module used for operations on 4D images.
    * :mod:`petpal.preproc.motion_corr` - module for motion correction tools.
    * :mod:`petpal.preproc.register` - module for MRI and atlas registration.

To do:
    * Fix docs
"""
import argparse
from ..utils import useful_functions
from ..preproc import image_operations_4d, motion_corr


_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Weighted Sum:
    petpal-preproc-2 weighted-series-sum -i /path/to/pet.nii -o /path/to/output.nii --half-life 6586.26
  - Auto Crop:
    petpal-preproc-2 auto-crop -i /path/to/pet.nii -o /path/to/output.nii -t 0.01
  - Motion Correction:
    petpal-preproc-2 motion-correction -i /path/to/pet.nii -o /path/to/output.nii --motion-target 0 600 --half-life 6586.26 --transform-type Rigid
""")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds common arguments ('--input-img', '--out-img', '--verbose') to a provided ArgumentParser object.

    This function modifies the passed ArgumentParser object by adding three arguments commonly used in the script.
    It uses the add_argument method of the ArgumentParser class. After running this function, the parser will
    be able to accept and parse these additional arguments from the command line when run.

    .. note::
        This function modifies the passed `parser` object in-place and does not return anything.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments are added.

    Raises:
        argparse.ArgumentError: If a duplicate argument tries to be added.

    Side Effects:
        Modifies the ArgumentParser object by adding new arguments.

    Example:
        .. code-block:: python

            parser = argparse.ArgumentParser()
            _add_common_args(parser)
            args = parser.parse_args(['--pet', 'pet_file', '--out-dir', './', '--prefix', 'prefix'])
            print(args.pet)
            print(args.out_dir)
            print(args.prefix)

    """
    parser.add_argument('-o', '--out-img', default='petpal_wss_output.nii.gz', help='Output image filename')
    parser.add_argument('-i', '--input-img',required=True,help='Path to input image.',type=str)
    parser.add_argument('-v', '--verbose', action='store_true',
                            help='Print processing information during computation.', required=False)


def _generate_args() -> argparse.Namespace:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.Namespace): Arguments used in the command line and their corresponding values.
    """
    parser = argparse.ArgumentParser(prog='petpal-preproc-2',
                                     description='Command line interface for running PET pre-processing steps.',
                                     epilog=_PREPROC_EXAMPLES_, formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")

    parser_wss = subparsers.add_parser('weighted-series-sum', help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_wss)
    parser_wss.add_argument('--half-life', required=True, help='Half life of radioisotope in seconds.',
                            type=float)
    parser_wss.add_argument('--start-time', required=False, help='Start time of sum in seconds.',
                            type=float,default=0)
    parser_wss.add_argument('--end-time', required=False, help='End time of sum in seconds.',
                            type=float,default=-1)

    parser_crop = subparsers.add_parser('auto-crop',
                                        help='Automatically crop 4D PET image using threshold.')
    _add_common_args(parser_crop)
    parser_crop.add_argument('-t','--thresh-val', required=True,default=0.01,
                            help='Fractional threshold to crop image projections.',type=float)


    parser_moco = subparsers.add_parser('motion-correction',
                                        help='Motion correct 4D PET data.')
    _add_common_args(parser_moco)
    parser_moco.add_argument('--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple (i.e. '-t 0 600' for first ten minutes).",
                            required=True)
    parser_moco.add_argument('--transform-type', required=False,default='Rigid',
                             help='Transformation type (Rigid or Affine).',type=str)
    parser_moco.add_argument('--half-life', required=False, 
                             help='Half life of radioisotope in seconds.'
                                  'Required for some motion targets.',type=float)

    return parser


def main():
    """
    Preproc command line interface
    """
    preproc_parser = _generate_args()
    args = preproc_parser.parse_args()

    if args.command is None:
        preproc_parser.print_help()
        raise SystemExit('Exiting without command')



    command = str(args.command).replace('-','_')

    if args.verbose:
        print(f"Running {command} with parameters")

    if command=='weighted_series_sum':
        useful_functions.weighted_series_sum(input_image_4d_path=args.input_img,
                                             out_image_path=args.out_img,
                                             half_life=args.half_life,
                                             start_time=args.start_time,
                                             end_time=args.end_time,
                                             verbose=args.verbose)

    if command=='auto_crop':
        image_operations_4d.SimpleAutoImageCropper(input_image_path=args.input_img,
                                                   out_image_path=args.out_img,
                                                   thresh_val=args.thresh_val,
                                                   verbose=args.verbose)

    if command=='motion_correction':
        if len(args.motion_target)==1:
            motion_target = args.motion_target[0]
        else:
            motion_target = args.motion_target
        motion_corr.motion_corr(input_image_4d_path=args.input_img,
                                out_image_path=args.out_img,
                                motion_target_option=motion_target,
                                verbose=True,
                                type_of_transform=args.transform_type,
                                half_life=args.half_life)

if __name__ == "__main__":
    main()
