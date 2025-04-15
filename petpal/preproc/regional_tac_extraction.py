"""
Regional TAC extraction
"""
import os
import nibabel
import numpy as np
from .image_operations_4d import extract_mean_roi_tac_from_nifti_using_segmentation
from ..utils import image_io


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
               time_frame_keyword: str = 'FrameReferenceTime',
               out_tac_prefix: str = '', ):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifti_with_same_filename(input_image_path)
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
    pet_numpy = nibabel.load(input_image_path).get_fdata()
    seg_numpy = nibabel.load(segmentation_image_path).get_fdata()

    for i, _maps in enumerate(label_map['mapping']):
        extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(regions_map[i]),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir, f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')
