"""This module contains all constants used throughout the library.

PET radionuclide half life source: code borrowed from DynamicPET
(https://github.com/bilgelm/dynamicpet/blob/main/src/dynamicpet/petbids/petbidsjson.py), derived
from TPC (turkupetcentre.net/petanalysis/decay.html). This source is from:
Table of Isotopes, Sixth edition, edited by C.M. Lederer, J.M. Hollander, I. Perlman. WILEY, 1967.
"""

HALF_LIVES = {
    "c11": 1224,
    "n13": 599,
    "o15": 123,
    "f18": 6588,
    "cu62": 582,
    "cu64": 45721.1,
    "ga68": 4080,
    "ge68": 23760000,
    "br76": 58700,
    "rb82": 75,
    "zr89": 282240,
    "i124": 360806.4,
}


CONVERT_kBq_to_mCi_ = 37000.0
r"""float: Convert kBq/ml to mCi/ml. 37000 kBq = 1 mCi.
"""