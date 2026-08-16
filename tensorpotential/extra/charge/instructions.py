"""FiLM conditioning of scalar node features on a per-structure total charge.

    P <- (1 + gamma(Q)) * P + beta(Q)

where Q is one scalar per structure, broadcast to that structure's atoms. This
mirrors the charge conditioning in LOREM so results transfer between the two
codebases.

PLACEMENT
---------
This is applied to the *scalar* node features immediately upstream of the
energy readout -- `rho` in GRACE-1L, and both `I_out_0_LN` and `I_1_LN` in
GRACE-2L (the readout sums both branches, so conditioning only one would leave
the other's energy contribution charge-independent). Everything stays on the
invariant path, so there is no equivariance constraint to respect.

Note the resulting model is still charge-sensitive in a useful way:
E_i = MLP((1 + gamma(Q)) * rho_i(geometry) + beta(Q)), and rho_i depends on
geometry, so d2E/drdq is nonzero and Born effective charges remain available.

IDENTITY AT Q = 0 AND AT INITIALISATION
---------------------------------------
Two independent guarantees, both of which matter for backwards compatibility:

1. `use_bias=False` in the gamma/beta MLP means gamma(0) = beta(0) = 0
   exactly, so an *uncharged* structure reproduces the unconditioned model no
   matter what the weights are.
2. `self.gate` is initialised to zeros (the same trick
   `InvariantLayerRMSNorm(init="zeros")` uses), so at step 0 FiLM is the
   identity for *every* Q. A FiLM preset loaded with pretrained base weights
   therefore starts as an exact copy of the base model, which is what makes
   finetuning a foundation model safe.
"""

from __future__ import annotations

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.functions.nn import FullyConnectedMLP
from tensorpotential.instructions.base import TPInstruction, capture_init_args


@capture_init_args
class FiLMChargeScalar(TPInstruction):
    """FiLM-condition a scalar node-feature tensor on the total charge.

    Parameters
    ----------
    inpt : TPInstruction
        Upstream instruction producing `[n_atoms, n_out, 1]` scalar features.
    normalize : {"none", "per_atom", "per_area"}
        What the gamma/beta MLPs actually see. "none" is the raw total charge,
        matching LOREM. "per_atom" is Q/N, which is size-extensive. "per_area"
        is Q/A, the surface charge density, which is the physically right
        conditioner for a slab.
    slab_normal_axis : int
        Index of the cell vector along the slab normal. **Only used by
        `normalize="per_area"`, and deliberately explicit rather than inferred
        from the cell.** Inferring it is exactly how a silent 2.54x error in
        Born effective charges arose in a sibling project: the surface of a
        slab whose normal is `a` is spanned by `b` and `c`, not by `a` and `b`.
    modulate_linear_channel : bool
        Whether FiLM also touches channel 0.

        `LinMLPOut2ScalarTarget` computes `E_i = rho_0 + MLP(rho_1..rho_n)`,
        and every `rho_k` is a learned linear contraction of the ACE basis
        (`FunctionReduceN`), so **channel 0 is a linear ACE energy**,
        sum_v c_v B_v. Routing it around the MLP keeps linear ACE as an exact
        subspace of the model, which is where ACE's systematic improvability
        comes from; the MLP then adds a correction rather than being the whole
        model.

        True (default): the summed `beta_0` term contributes `N * beta_0(Q)`, a
        size-extensive purely charge-dependent offset -- i.e. the capacitive
        q^2/2C energy, which this data demonstrably has (cpmace's d2E/dq2 is
        +7.68 eV and near-constant across geometries). Reaching that through
        the MLP branch alone is possible but indirect and entangled with the
        geometry. `(1 + gamma_0)` additionally makes the linear ACE
        coefficients charge-dependent, i.e. bonding responds to charge.

        False: the linear ACE path stays exactly charge-independent, giving a
        clean "charge-free linear reference + charge-dependent corrections"
        split. Worth trying when **finetuning a pretrained foundation model**,
        where rho_0 carries most of the energy and multiplying it by a growing
        (1 + gamma_0) is the most destabilising thing FiLM could do early.

        Note the two GRACE-2L branches are not symmetric here. `I_out_0_LN`
        uses `InvariantLayerRMSNorm(type="only_nonlin")`, which passes channel
        0 through completely untouched -- a pristine linear ACE term. `I_1_LN`
        uses `type="full"`, which RMS-normalises *all* channels including 0, so
        its "linear" channel is already per-atom rescaled and is not a pure
        linear ACE energy. This flag therefore means something slightly
        different on each branch.
    """

    def __init__(
        self,
        inpt: TPInstruction,
        name: str = "FiLMChargeScalar",
        hidden_layers: list[int] = None,
        activation: str = "silu",
        normalize: str = "none",
        slab_normal_axis: int = 2,
        modulate_linear_channel: bool = True,
        **kwargs,
    ):
        super().__init__(name=name)
        self.input = inpt
        # mirrored so downstream consumers see an unchanged interface:
        # LinMLPOut2ScalarTarget asserts max(origin.lmax) == 0 and that all
        # origins share n_out.
        self.n_out = inpt.n_out
        self.lmax = 0
        # this instruction always emits [n_atoms, n_out, lm], never lm-first,
        # regardless of what the input used
        self.lm_first = False

        assert normalize in ("none", "per_atom", "per_area"), (
            f"unknown normalize={normalize!r}, expected none/per_atom/per_area"
        )
        assert slab_normal_axis in (0, 1, 2), "slab_normal_axis must be 0, 1 or 2"
        self.normalize = normalize
        self.slab_normal_axis = slab_normal_axis
        self.modulate_linear_channel = modulate_linear_channel
        self.hidden_layers = [16] if hidden_layers is None else hidden_layers
        self.activation = activation

        # Instance attribute, not class attribute: `per_area` needs the cell
        # and the others must not request it. TPModel.build only does
        # hasattr(sm, "input_tensor_spec"), so an instance attribute works and
        # keeps the tf.function signature as narrow as possible.
        self.input_tensor_spec = {
            constants.TOTAL_CHARGE: {"shape": [None, 1], "dtype": "float"},
            constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        }
        if self.normalize == "per_atom":
            self.input_tensor_spec[constants.N_STRUCTURES_BATCH_TOTAL] = {
                "shape": [],
                "dtype": "int",
            }
        elif self.normalize == "per_area":
            self.input_tensor_spec[constants.CELL_VECTORS] = {
                "shape": [None, 3, 3],
                "dtype": "float",
            }

        self.mlp = FullyConnectedMLP(
            input_size=1,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            output_size=2 * self.n_out,
            # no biases => gamma(0) = beta(0) = 0, so a neutral structure is
            # bit-identical to the unconditioned model
            use_bias=False,
            name=self.name + "_MLP",
        )

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.mlp.is_built:
            self.mlp.build(float_dtype)
        # zero-init => FiLM is the identity at step 0 for every Q
        self.gate = tf.Variable(
            tf.zeros([1, 2 * self.n_out], dtype=float_dtype), name="gate"
        )
        self.is_built = True

    def _structure_charge(self, input_data):
        """Per-structure conditioning scalar, shape [n_struct, 1]."""
        q = input_data[constants.TOTAL_CHARGE]

        if self.normalize == "per_atom":
            map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP]
            nat = tf.math.unsorted_segment_sum(
                tf.ones_like(map_at2struc, dtype=q.dtype),
                map_at2struc,
                num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
            )
            # padded structures can have zero atoms; guard the divide
            nat = tf.reshape(nat, [-1, 1])
            return tf.math.divide_no_nan(q, nat)

        if self.normalize == "per_area":
            cell = input_data[constants.CELL_VECTORS]
            i, j = [k for k in range(3) if k != self.slab_normal_axis]
            area = tf.linalg.norm(
                tf.linalg.cross(cell[:, i, :], cell[:, j, :]), axis=-1
            )
            return tf.math.divide_no_nan(q, tf.reshape(area, [-1, 1]))

        return q

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        x = input_data[self.input.name]
        if getattr(self.input, "lm_first", False):
            # [lm, atoms, n_out] -> [atoms, n_out, lm]
            x = tf.transpose(x, [1, 2, 0])

        q = tf.cast(self._structure_charge(input_data), x.dtype)
        # broadcast the structure's charge to each of its atoms. Padded atoms
        # are mapped to the last (dummy) structure slot by the data builder's
        # pad_batch, so this gather is always in range.
        q_at = tf.gather(q, input_data[constants.ATOMS_TO_STRUCTURE_MAP])

        gamma_beta = self.mlp(q_at) * tf.cast(self.gate, x.dtype)
        gamma = gamma_beta[:, : self.n_out, None]
        beta = gamma_beta[:, self.n_out :, None]

        if not self.modulate_linear_channel:
            # leave channel 0 -- the linear passthrough -- exactly as it was
            mask = tf.concat(
                [
                    tf.zeros([1, 1, 1], dtype=x.dtype),
                    tf.ones([1, self.n_out - 1, 1], dtype=x.dtype),
                ],
                axis=1,
            )
            gamma = gamma * mask
            beta = beta * mask

        return x * (1.0 + gamma) + beta
