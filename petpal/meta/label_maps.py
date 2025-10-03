"""Label maps stored as dictionaries.

This module contains label maps for use in running PETPAL.
"""
from collections.abc import MutableSequence, Callable
import pathlib
from numbers import Integral
from petpal.utils.image_io import safe_load_meta
from petpal.utils.useful_functions import str_to_camel_case

label_map_freesurfer = {
    'Unknown': 0,
    'LeftCerebralWhiteMatter': 2,
    'LeftLateralVentricle': 4,
    'LeftInfLatVent': 5,
    'LeftCerebellumWhiteMatter': 7,
    'LeftCerebellumCortex': 8,
    'LeftThalamus': 10,
    'LeftCaudate': 11,
    'LeftPutamen': 12,
    'LeftPallidum': 13,
    '3rdVentricle': 14,
    '4thVentricle': 15,
    'BrainStem': 16,
    'LeftHippocampus': 17,
    'LeftAmygdala': 18,
    'CSF': 24,
    'LeftAccumbensArea': 26,
    'LeftVentralDC': 28,
    'LeftVessel': 30,
    'LeftChoroidPlexus': 31,
    'RightCerebralWhiteMatter': 41,
    'RightLateralVentricle': 43,
    'RightInfLatVent': 44,
    'RightCerebellumWhiteMatter': 46,
    'RightCerebellumCortex': 47,
    'RightThalamus': 49,
    'RightCaudate': 50,
    'RightPutamen': 51,
    'RightPallidum': 52,
    'RightHippocampus': 53,
    'RightAmygdala': 54,
    'RightAccumbensArea': 58,
    'RightVentralDC': 60,
    'RightVessel': 62,
    'RightChoroidPlexus': 63,
    'OpticChiasm': 85,
    'AirCavity': 130,
    'Skull': 165,
    'Vermis': 172,
    'Midbrain': 173,
    'Pons': 174,
    'Medulla': 175,
    'CCPosterior': 251,
    'CCMidPosterior': 252,
    'CCCentral': 253,
    'CCMidAnterior': 254,
    'CCAnterior': 255,
    'CSFExtraCerebral': 257,
    'HeadExtraCerebral': 258,
    'CtxLhBankssts': 1001,
    'CtxLhCaudalanteriorcingulate': 1002,
    'CtxLhCaudalmiddlefrontal': 1003,
    'CtxLhCuneus': 1005,
    'CtxLhEntorhinal': 1006,
    'CtxLhFusiform': 1007,
    'CtxLhInferiorparietal': 1008,
    'CtxLhInferiortemporal': 1009,
    'CtxLhIsthmuscingulate': 1010,
    'CtxLhLateraloccipital': 1011,
    'CtxLhLateralorbitofrontal': 1012,
    'CtxLhLingual': 1013,
    'CtxLhMedialorbitofrontal': 1014,
    'CtxLhMiddletemporal': 1015,
    'CtxLhParahippocampal': 1016,
    'CtxLhParacentral': 1017,
    'CtxLhParsopercularis': 1018,
    'CtxLhParsorbitalis': 1019,
    'CtxLhParstriangularis': 1020,
    'CtxLhPericalcarine': 1021,
    'CtxLhPostcentral': 1022,
    'CtxLhPosteriorcingulate': 1023,
    'CtxLhPrecentral': 1024,
    'CtxLhPrecuneus': 1025,
    'CtxLhRostralanteriorcingulate': 1026,
    'CtxLhRostralmiddlefrontal': 1027,
    'CtxLhSuperiorfrontal': 1028,
    'CtxLhSuperiorparietal': 1029,
    'CtxLhSuperiortemporal': 1030,
    'CtxLhSupramarginal': 1031,
    'CtxLhFrontalpole': 1032,
    'CtxLhTemporalpole': 1033,
    'CtxLhTransversetemporal': 1034,
    'CtxLhInsula': 1035,
    'CtxRhBankssts': 2001,
    'CtxRhCaudalanteriorcingulate': 2002,
    'CtxRhCaudalmiddlefrontal': 2003,
    'CtxRhCuneus': 2005,
    'CtxRhEntorhinal': 2006,
    'CtxRhFusiform': 2007,
    'CtxRhInferiorparietal': 2008,
    'CtxRhInferiortemporal': 2009,
    'CtxRhIsthmuscingulate': 2010,
    'CtxRhLateraloccipital': 2011,
    'CtxRhLateralorbitofrontal': 2012,
    'CtxRhLingual': 2013,
    'CtxRhMedialorbitofrontal': 2014,
    'CtxRhMiddletemporal': 2015,
    'CtxRhParahippocampal': 2016,
    'CtxRhParacentral': 2017,
    'CtxRhParsopercularis': 2018,
    'CtxRhParsorbitalis': 2019,
    'CtxRhParstriangularis': 2020,
    'CtxRhPericalcarine': 2021,
    'CtxRhPostcentral': 2022,
    'CtxRhPosteriorcingulate': 2023,
    'CtxRhPrecentral': 2024,
    'CtxRhPrecuneus': 2025,
    'CtxRhRostralanteriorcingulate': 2026,
    'CtxRhRostralmiddlefrontal': 2027,
    'CtxRhSuperiorfrontal': 2028,
    'CtxRhSuperiorparietal': 2029,
    'CtxRhSuperiortemporal': 2030,
    'CtxRhSupramarginal': 2031,
    'CtxRhFrontalpole': 2032,
    'CtxRhTemporalpole': 2033,
    'CtxRhTransversetemporal': 2034,
    'CtxRhInsula': 2035
    }


label_map_freesurfer_merge_lr = {
    'Unknown': 0,
    'CerebralWhiteMatter': [2, 41],
    'LateralVentricle': [4, 43],
    'InfLatVent': [5, 44],
    'CerebellumWhiteMatter': [7, 46],
    'CerebellumCortex': [8, 47],
    'Thalamus': [10, 49],
    'Caudate': [11, 50],
    'Putamen': [12, 51],
    'Pallidum': [13, 52],
    '3rdVentricle': 14,
    '4thVentricle': 15,
    'BrainStem': 16,
    'Hippocampus': [17, 53],
    'Amygdala': [18, 54],
    'CSF': 24,
    'AccumbensArea': [26, 58],
    'VentralDC': [28, 60],
    'Vessel': [30, 62],
    'ChoroidPlexus': [31, 63],
    'OpticChiasm': 85,
    'AirCavity': 130,
    'Skull': 165,
    'Vermis': 172,
    'Midbrain': 173,
    'Pons': 174,
    'Medulla': 175,
    'CCPosterior': 251,
    'CCMidPosterior': 252,
    'CCCentral': 253,
    'CCMidAnterior': 254,
    'CCAnterior': 255,
    'CSFExtraCerebral': 257,
    'HeadExtraCerebral': 258,
    'CtxBankssts': [1001, 2001],
    'CtxCaudalanteriorcingulate': [1002, 2002],
    'CtxCaudalmiddlefrontal': [1003, 2003],
    'CtxCuneus': [1005, 2005],
    'CtxEntorhinal': [1006, 2006],
    'CtxFusiform': [1007, 2007],
    'CtxInferiorparietal': [1008, 2008],
    'CtxInferiortemporal': [1009, 2009],
    'CtxIsthmuscingulate': [1010, 2010],
    'CtxLateraloccipital': [1011, 2011],
    'CtxLateralorbitofrontal': [1012, 2012],
    'CtxLingual': [1013, 2013],
    'CtxMedialorbitofrontal': [1014, 2014],
    'CtxMiddletemporal': [1015, 2015],
    'CtxParahippocampal': [1016, 2016],
    'CtxParacentral': [1017, 2017],
    'CtxParsopercularis': [1018, 2018],
    'CtxParsorbitalis': [1019, 2019],
    'CtxParstriangularis': [1020, 2020],
    'CtxPericalcarine': [1021, 2021],
    'CtxPostcentral': [1022, 2022],
    'CtxPosteriorcingulate': [1023, 2023],
    'CtxPrecentral': [1024, 2024],
    'CtxPrecuneus': [1025, 2025],
    'CtxRostralanteriorcingulate': [1026, 2026],
    'CtxRostralmiddlefrontal': [1027, 2027],
    'CtxSuperiorfrontal': [1028, 2028],
    'CtxSuperiorparietal': [1029, 2029],
    'CtxSuperiortemporal': [1030, 2030],
    'CtxSupramarginal': [1031, 2031],
    'CtxFrontalpole': [1032, 2032],
    'CtxTemporalpole': [1033, 2033],
    'CtxTransversetemporal': [1034, 2034],
    'CtxInsula': [1035, 2035]
    }


class LabelMapLoader:
    """Load label map data"""
    def __init__(self, label_map_option: str | dict):
        self.loader_method = self.detect_option(label_map_option=label_map_option)
        self.label_map = self.loader_method(label_map_option)
        self.labels_to_camel_case()
        self.validate_mappings()

    def from_petpal(self, label_map_name: str) -> dict:
        """Choose from an existing list of label maps implemented in PETPAL."""
        match label_map_name.lower():
            case 'freesurfer':
                return label_map_freesurfer
            case 'freesurfer_merge_lr':
                return label_map_freesurfer_merge_lr
            case _:
                raise ValueError(f"Label map name {label_map_name} not in existing list of "
                                 "implemented label maps. Choose one of: 'freesurfer' or "
                                 "'freesurfer_merge_lr'.")

    def from_dict(self, label_map: dict) -> dict:
        """Provide a label map implemented in Python."""
        return label_map

    def from_json(self, label_map_path: str) -> dict:
        """Load a label map from a .json file."""
        return safe_load_meta(input_metadata_file=label_map_path)

    def detect_option(self, label_map_option: dict | str) -> Callable:
        """Determine the label map loading method to use based on the provided option."""
        if isinstance(label_map_option, dict):
            return self.from_dict
        if isinstance(label_map_option, str):
            label_map_path = pathlib.Path(label_map_option)
            if label_map_path.exists():
                return self.from_json
            elif label_map_path.suffix!='':
                raise FileNotFoundError(f'Label map option {label_map_option} looks like a path'
                                        'yet does not exist.')
            return self.from_petpal

    def labels_to_camel_case(self):
        """Convert all label map labels to camel case and update label map."""
        label_map = self.label_map.copy()
        labels = label_map.keys()
        for label in labels:
            updated_label = str_to_camel_case(label)
            self.label_map[updated_label] = self.label_map.pop(label)

    def validate_mappings(self):
        """Validate mapping values for integer mappings in the label map. Mappings can be an
        integer or a list of integers.
        
        Raises:
            """
        label_map = self.label_map.copy()
        labels = label_map.keys()
        mappings = label_map.values()
        for label, mapping in zip(labels, mappings):
            if isinstance(mapping, Integral):
                continue
            if isinstance(mapping, MutableSequence):
                for value in mapping:
                    if isinstance(value, Integral):
                        continue
                    raise TypeError(f'Label {label} with mapping {mapping} contains value {value} '
                                    f'which is not an integer. Instead found type: {type(value)}.')
            else:
                raise TypeError(f'Label {label} contains mapping {mapping} which is not '
                                f'an integer or a list. Instead found type: {type(mapping)}.')
