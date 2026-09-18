"""Losses on the charge derivatives of the energy.

- `WeightedWorkFunctionLoss` -- per structure, on dE/dq.
- `WeightedDFDQLoss`         -- per atom and component, on dF/dq (the Born
                                effective charges, in raw dF/dq units).
- `WeightedD2Edq2Loss`       -- per structure, on d2E/dq2.

The last two need `predict_bec=True` on the compute function; the first does
not. Enable in input.yaml with::

    fit:
      loss:
        extra_components:
          WeightedWorkFunctionLoss: {weight: 0.05}
          WeightedDFDQLoss: {weight: 1.0}
"""

import tensorflow as tf

from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.metrics import (
    D2Edq2Metrics,
    DFDQMetrics,
    WorkFunctionMetrics,
)
from tensorpotential.loss import LossComponent, huber


class WeightedWorkFunctionLoss(LossComponent):
    """Weighted squared or huber error on dE/dq.

    Named for the target, not the loss shape, unlike GRACE's core classes
    (`WeightedSSEForceLoss` vs `WeightedHuberForceLoss` are separate classes
    picked by a `type:` dispatch dict). This one takes `type` as a constructor
    argument instead, because the `extra_components` path resolves a class by
    name with no dispatch dict -- so a shape-per-class split would buy the YAML
    nothing. An `SSE` prefix would then be a lie whenever `type: huber` is set.

    Per-structure, so this is shaped exactly like the virial loss rather than
    the force loss.
    """

    input_tensor_spec = {
        cc.DATA_REFERENCE_WORK_FUNCTION: {"shape": [None, 1], "dtype": "float"},
        cc.DATA_WORK_FUNCTION_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedWorkFunctionLoss",
        type: str = "square",
        delta: float = 0.1,
        normalize_by_samples: bool = True,
        **kwargs,
    ):
        super(WeightedWorkFunctionLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.corresponding_metrics = WorkFunctionMetrics
        assert type in ["square", "huber"]
        self.type = type
        self.delta = delta

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        wf_true = input_data[cc.DATA_REFERENCE_WORK_FUNCTION]
        wf_weight = input_data[cc.DATA_WORK_FUNCTION_WEIGHTS]
        wf_pred = predictions[cc.PREDICT_WORK_FUNCTION]

        d_wf = wf_pred - wf_true
        if self.type == "huber":
            err = huber(d_wf, delta=self.delta)
        else:
            err = tf.square(d_wf)

        # padded structures carry weight 0, so they drop out here
        loss_wf = tf.reduce_sum(wf_weight * err)
        if self.normalize_by_samples:
            loss_wf /= tf.reduce_sum(wf_weight) + self.epsilon
        return loss_wf


class WeightedDFDQLoss(LossComponent):
    """Weighted squared or huber error on dF/dq, per atom and component.

    Shaped like the force loss rather than the virial loss: the label is
    [n_atoms, 3] and padded atoms carry weight 0, so they drop out of both the
    numerator and the denominator.

    The units are raw dF/dq, eV/(A e). `databuilder.py` converts the
    dimensionless Born form Z* = (A eps0) dF/dq once, on the way in.
    """

    input_tensor_spec = {
        cc.DATA_REFERENCE_DF_DQ: {"shape": [None, 3], "dtype": "float"},
        cc.DATA_DF_DQ_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedDFDQLoss",
        type: str = "square",
        delta: float = 0.01,
        normalize_by_samples: bool = True,
        **kwargs,
    ):
        super(WeightedDFDQLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.corresponding_metrics = DFDQMetrics
        assert type in ["square", "huber"]
        self.type = type
        self.delta = delta

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        true = input_data[cc.DATA_REFERENCE_DF_DQ]
        weight = input_data[cc.DATA_DF_DQ_WEIGHTS]
        pred = predictions[cc.PREDICT_DF_DQ]

        d = pred - true
        err = huber(d, delta=self.delta) if self.type == "huber" else tf.square(d)

        loss = tf.reduce_sum(weight * tf.reduce_sum(err, axis=1, keepdims=True))
        if self.normalize_by_samples:
            loss /= 3.0 * tf.reduce_sum(weight) + self.epsilon
        return loss


class WeightedD2Edq2Loss(LossComponent):
    """Weighted squared or huber error on d2E/dq2, per structure.

    Free to compute -- it is the second source of the same gradient call that
    produces dF/dq -- but not free to train on: in the sibling LOREM
    experiments it cost ~10% on the forces. Off unless a weight is given.
    """

    input_tensor_spec = {
        cc.DATA_REFERENCE_D2E_DQ2: {"shape": [None, 1], "dtype": "float"},
        cc.DATA_D2E_DQ2_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedD2Edq2Loss",
        type: str = "square",
        delta: float = 0.1,
        normalize_by_samples: bool = True,
        **kwargs,
    ):
        super(WeightedD2Edq2Loss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.corresponding_metrics = D2Edq2Metrics
        assert type in ["square", "huber"]
        self.type = type
        self.delta = delta

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        true = input_data[cc.DATA_REFERENCE_D2E_DQ2]
        weight = input_data[cc.DATA_D2E_DQ2_WEIGHTS]
        pred = predictions[cc.PREDICT_D2E_DQ2]

        d = pred - true
        err = huber(d, delta=self.delta) if self.type == "huber" else tf.square(d)

        loss = tf.reduce_sum(weight * err)
        if self.normalize_by_samples:
            loss /= tf.reduce_sum(weight) + self.epsilon
        return loss
