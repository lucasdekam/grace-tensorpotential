"""Work-function loss: a per-structure weighted squared or huber error on dE/dq.

Enable in input.yaml with::

    fit:
      loss:
        extra_components:
          WeightedWorkFunctionLoss: {weight: 0.05}
"""

import tensorflow as tf

from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.metrics import WorkFunctionMetrics
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
