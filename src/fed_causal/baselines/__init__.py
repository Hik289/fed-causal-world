from .base import BaselineBase
from .b2_global_seq_wm import B2_GlobalSeqWM
from .others import (B0_NoWM, B1_LocalWM, B3_GlobalTransitionGraph, B4_CorrelationWM,
                     B5_TemporalWM, B6_InvariantWM, B7_CausalWMNoInt,
                     B8_FedCausalCompose, B9_FCCNoControl, B10_OracleCausalWM,
                     B11_CentralizedSeq)

REGISTRY = {
    'B0': B0_NoWM, 'B1': B1_LocalWM, 'B2': B2_GlobalSeqWM, 'B3': B3_GlobalTransitionGraph,
    'B4': B4_CorrelationWM, 'B5': B5_TemporalWM, 'B6': B6_InvariantWM,
    'B7': B7_CausalWMNoInt, 'B8': B8_FedCausalCompose, 'B9': B9_FCCNoControl,
    'B10': B10_OracleCausalWM, 'B11': B11_CentralizedSeq,
}
