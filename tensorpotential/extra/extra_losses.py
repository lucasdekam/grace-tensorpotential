"""Loss components gracemaker can name in `fit.loss.extra_components`.

`cli/prepare.py` resolves those names with `getattr` against this module, so a
class that is not re-exported here is invisible to a fit however complete it is
elsewhere.
"""

from tensorpotential.extra.gen_tensor.loss import WeightedTensorLoss
from tensorpotential.extra.charge.loss import (
    WeightedD2Edq2Loss,
    WeightedDFDQLoss,
    WeightedWorkFunctionLoss,
)

__all__ = [
    "WeightedTensorLoss",
    "WeightedWorkFunctionLoss",
    "WeightedDFDQLoss",
    "WeightedD2Edq2Loss",
]
