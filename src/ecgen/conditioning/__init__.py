"""
Conditioning: encode condition fields into :class:`ecgen.types.ConditionOutput`.

Importing this package registers the field encoders in
``ecgen.registry.CONDITION_ENCODERS``: ``categorical``, ``multilabel``,
``numeric``, ``fourier``, ``masked_signal``, ``text_hash``, ``text``.
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
from ecgen.conditioning.text import HashTextEncoder, PretrainedTextEncoder

__all__ = [
    "HashTextEncoder",
    "PretrainedTextEncoder",
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
