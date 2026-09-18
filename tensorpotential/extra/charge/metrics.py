"""Charge-derivative metrics, mirroring `extra/gen_tensor/metrics.py`.

Reported as `mae/wf` and `rmse/wf` (cli/metrics.py rewrites the abs/sqr keys).
Named for the quantity, not the residual: the `rmse/` prefix already says it is
an error, and a `d_` prefix would be actively misleading here because the work
function *is* a derivative, dE/dq -- `d_wf` reads as d(work function)/d(...).
This also matches `f_comp` / `virial` / `stress`, which are likewise residuals
named for the quantity; only `de` / `depa` carry a delta prefix.
"""

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
        err = wf_true - wf_pred

        return {
            "abs/wf/per_struct": tf.reduce_sum(tf.math.abs(err)),
            "sqr/wf/per_struct": tf.reduce_sum(err**2),
        }

    @property
    def normalization_spec(self) -> dict[str, dict]:
        return {
            "abs/wf/per_struct": {"norm": "n_structures", "factor": 1.0},
            "sqr/wf/per_struct": {"norm": "n_structures", "factor": 1.0},
        }


class DFDQMetrics(AbstractMetrics):
    """Per-component error on dF/dq, shaped like the force metrics.

    Reported in raw dF/dq units, eV/(A e) -- the same units the model predicts
    and the loss sees. The dimensionless Born form Z* = (A eps0) dF/dq differs
    by a per-dataset constant applied in `databuilder.py`, and converting back
    here would put a second area calculation in the codebase, which is the one
    thing that note forbids.
    """

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        cc.DATA_REFERENCE_DF_DQ: {"shape": [None, 3], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        n_real = input_data[constants.N_ATOMS_BATCH_REAL]
        true = input_data[cc.DATA_REFERENCE_DF_DQ][:n_real]
        pred = predictions[cc.PREDICT_DF_DQ][:n_real]
        err = true - pred

        # RMSE only. Each metric costs two numbers on every log line, and the
        # MAE of a derivative target adds little next to its RMSE -- `wf` keeps
        # both so the logs of runs that predate these stay comparable.
        return {"sqr/dfdq/per_struct": tf.reduce_sum(err**2)}

    @property
    def normalization_spec(self) -> dict[str, dict]:
        return {
            # `/per_struct` despite normalising by n_atoms, exactly as
            # ForceMetrics does: cli/metrics.py sums only keys with that
            # suffix, so anything else is dropped without a word.
            "sqr/dfdq/per_struct": {"norm": "n_atoms", "factor": 3.0},
        }


class D2Edq2Metrics(AbstractMetrics):
    """Per-structure error on d2E/dq2, in V/e. Shaped like the work function."""

    input_tensor_spec = {
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        cc.DATA_REFERENCE_D2E_DQ2: {"shape": [None, 1], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        n_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        true = input_data[cc.DATA_REFERENCE_D2E_DQ2][:n_real]
        pred = predictions[cc.PREDICT_D2E_DQ2][:n_real]
        err = true - pred

        return {"sqr/d2edq2/per_struct": tf.reduce_sum(err**2)}

    @property
    def normalization_spec(self) -> dict[str, dict]:
        return {
            "sqr/d2edq2/per_struct": {"norm": "n_structures", "factor": 1.0},
        }
