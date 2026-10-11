"""
Conditioning: encode condition fields into :class:`ecgen.types.ConditionOutput`.

Importing this package registers the field encoders in
``ecgen.registry.CONDITION_ENCODERS``: ``categorical``, ``multilabel``,
``numeric``, ``fourier``, ``masked_signal``.
"""

from ecgen.conditioning.composite import (
    OBS_FIELD,
    CompositeConditioner,
    build_conditioner,
)
from ecgen.conditioning.encoders import (
    CategoricalEncoder,
    FieldEncoder,
    FourierScalarEncoder,
    MaskedSignalEncoder,
    MultiLabelEncoder,
    NumericEncoder,
)

__all__ = [
    "OBS_FIELD",
    "CompositeConditioner",
    "build_conditioner",
    "FieldEncoder",
    "CategoricalEncoder",
    "MultiLabelEncoder",
    "NumericEncoder",
    "FourierScalarEncoder",
    "MaskedSignalEncoder",
]
