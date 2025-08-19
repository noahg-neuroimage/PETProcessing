"""Utilities for metadata handling, scrubbing, etc..."""

from shutil import copyfile
from copy import deepcopy
from itertools import accumulate

from .image_io import safe_load_meta, write_dict_to_json
from .constants import HALF_LIVES
from .scan_timing import calculate_frame_reference_time
from ..preproc.decay_correction import calculate_frame_decay_factor


class BidsMetadataMender:
    """Class for repairing and filling in the gaps of BIDS metadata based on existing fields.

    For most use cases, just initialize a mender object, passing a string path to the .json file,
    and a boolean for whether the image was decay_corrected. Then, simply calling the object 
    (i.e. 'mender()') will result in keys being added/updated where possible.

    The following checks are made: 

    - If 'FrameDuration' is present, fill 'FrameTimesStart', assuming the first entry to be 0.
    - If 'TracerRadionuclide' is present, add 'RadionuclideHalfLife'.
        - Note that 'RadionuclideHalfLife' is not listed in BIDS, but we find it useful to store.
    - If all the previous keys were added, use them to calculate 'FrameReferenceTime'.
        - 'FrameReferenceTime' is not BIDS-required but it is useful.
    - If all the previous keys were added, add 'DecayCorrectionFactor' and 'ImageDecayCorrected'.
        - If decay_correction was set to False, 'DecayCorrectionFactor' is a list of ones of 
          len(FrameDuration) and 'ImageDecayCorrected will be 'false', per BIDS.

    Attributes: 
        metadata (dict): dictionary containing all the existing BIDS metadata
        filepath (str): path to .json file containing original BIDS metadata
        decay_correction (bool): whether or not the image is decay-corrected
    """
    metadata: dict
    filepath: str
    decay_correction: bool

    def __init__(self, json_filepath: str, decay_correction: bool = False):
        """Initializes a new BidsMetadataMender object.
        
        Args: 
            json_filepath (str): path to .json file with original BIDS metadata.
            decay_correction (bool): Whether or not the image is decay-corrected. Defaults to False.
        """
        self.metadata = safe_load_meta(input_metadata_file=json_filepath)
        self.filepath = json_filepath
        self.decay_correction = decay_correction

    def __call__(self, output_filepath: str | None = None):
        self._add_missing_keys()
        self._to_file(output_filepath)

    def _add_missing_keys(self):
        """Repair/Fill missing keys where possible."""
        updated_keys = []
        if 'FrameDuration' in self.metadata:
            self._add_frame_times_start()
            updated_keys.append('FrameTimesStart')
        if 'TracerRadionuclide' in self.metadata:
            self._add_half_life()
            updated_keys.append('RadionuclideHalfLife')
        if {'RadionuclideHalfLife', 'FrameDuration', 'FrameTimesStart'}.issubset(self.metadata):
            self._add_frame_reference_times()
            updated_keys.append('FrameReferenceTime')
        if self.decay_correction and {'RadionuclideHalfLife', 'FrameReferenceTime'}.issubset(self.metadata):
            self._add_decay_factors()
            updated_keys += ['DecayCorrectionFactor', 'ImageDecayCorrected']
        else:
            self._add_empty_decay_factors()
            updated_keys += ['DecayCorrectionFactor', 'ImageDecayCorrected']
        print(f'The following keys were updated: {updated_keys}')

    def _add_half_life(self):
        """Add "RadionuclideHalfLife" key to metadata."""
        metadata = deepcopy(self.metadata)
        radionuclide = metadata['TracerRadionuclide'].lower().replace("-", "")
        half_life = float(HALF_LIVES[radionuclide])
        metadata['RadionuclideHalfLife'] = half_life
        self.metadata = metadata

    def _add_empty_decay_factors(self):
        """Adds a list of ones for decay factors and sets 'ImageDecayCorrected' to False."""
        metadata = deepcopy(self.metadata)
        frame_durations = metadata['FrameDuration']
        decay_factors = [1 for i in frame_durations]
        metadata['DecayCorrectionFactor'] = decay_factors
        metadata['ImageDecayCorrected'] = 'False'
        self.metadata = metadata

    def _add_decay_factors(self):
        """Computes decay factors and adds 'DecayCorrectionFactor' to metadata."""
        metadata = deepcopy(self.metadata)
        half_life = metadata['RadionuclideHalfLife']
        decay_factors = [calculate_frame_decay_factor(frame_reference_time=t, half_life=half_life) for t in metadata['FrameReferenceTime']]
        metadata['DecayCorrectionFactor'] = decay_factors
        metadata.pop('DecayFactor', None)
        metadata['ImageDecayCorrected'] = 'True'
        self.metadata = metadata

    def _add_frame_times_start(self):
        """Fill in frame starts from frame durations, assuming first frame starts at 0."""
        metadata = deepcopy(self.metadata)
        frame_durations = metadata['FrameDuration']
        frame_starts = [0]
        frame_starts = frame_starts + list(accumulate(frame_durations[:-1]))
        metadata['FrameTimesStart'] = frame_starts
        self.metadata = metadata

    def _add_frame_reference_times(self):
        """Fill in frame reference times from frame starts and durations."""
        metadata = deepcopy(self.metadata)
        half_life = metadata['RadionuclideHalfLife']
        frame_starts = metadata['FrameTimesStart']
        frame_durations = metadata['FrameDuration']
        frame_reference_times = [calculate_frame_reference_time(frame_duration=duration, frame_start=start, half_life=half_life) for start, duration in zip(frame_starts, frame_durations)]
        metadata['FrameReferenceTime'] = frame_reference_times
        self.metadata = metadata

    def _to_file(self, filepath: str | None = None):
        """Write metadata dictionary to a .json file; defaults to making a backup."""
        if filepath is None:
            filepath = self.filepath
            copyfile(src=filepath, dst=filepath+".bak")
        write_dict_to_json(meta_data_dict=self.metadata, out_path=filepath)
