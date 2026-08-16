"""Work-function metrics, mirroring `extra/gen_tensor/metrics.py`."""

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.extra.charge import constants as cc
from tensorpotential.metrics import AbstractMetrics


class WorkFunctionMetrics(AbstractMetrics):
    input_tensor_spec = {
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        cc.DATA_REFERENCE_WORK_FUNCTION: {"shape": [None, 1], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        n_struct_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        # slice off the padded structures rather than relying on their labels
        # being zero -- the prediction there is meaningless, not zero
        wf_true = input_data[cc.DATA_REFERENCE_WORK_FUNCTION][:n_struct_real]
        wf_pred = predictions[cc.PREDICT_WORK_FUNCTION][:n_struct_real]
        d_wf = wf_true - wf_pred

        return {
            "abs/d_wf/per_struct": tf.reduce_sum(tf.math.abs(d_wf)),
            "sqr/d_wf/per_struct": tf.reduce_sum(d_wf**2),
        }

    @property
    def normalization_spec(self) -> dict[str, dict]:
        return {
            "abs/d_wf/per_struct": {"norm": "n_structures", "factor": 1.0},
            "sqr/d_wf/per_struct": {"norm": "n_structures", "factor": 1.0},
        }
