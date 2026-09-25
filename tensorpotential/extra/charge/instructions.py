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
from tensorpotential.extra.charge import constants as cc
from tensorpotential.functions.nn import FullyConnectedMLP
from tensorpotential.instructions.base import TPInstruction, capture_init_args
from tensorpotential.instructions.output import CreateOutputTarget, TPOutputInstruction


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
            # needed to zero gamma/beta on padded atoms -- see frwrd
            constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
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

        # Zero gamma and beta on PADDED atoms. Without this, FiLM breaks an
        # invariant the rest of GRACE relies on: an isolated (fake) atom has all
        # descriptors zero, and every MLP here is use_bias=False, so zero
        # features give zero energy. FiLM is affine, so a padded atom instead
        # gets 0*(1+gamma) + beta = beta != 0 -- and since padded atoms gather
        # the REAL structure's charge via map_atoms_to_structure, that beta
        # varies with q and leaks into both the energy and dE/dq.
        #
        # Verified: stock GRACE (GRACE-1L-OAM through TPCalculator) gives padded
        # atoms exactly 0.000000 eV at any pad_atoms_number, while our FiLM
        # models gave -0.117 eV and a 0.110 V shift in dE/dq. This is ours, not
        # upstream.
        real = tf.reshape(
            tf.range(tf.shape(gamma_beta)[0], dtype=tf.int32), [-1, 1]
        ) < input_data[constants.N_ATOMS_BATCH_REAL]
        gamma_beta = tf.where(real, gamma_beta, tf.zeros_like(gamma_beta))

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


@capture_init_args
class HelmholtzChargeTarget(TPOutputInstruction):
    r"""Impose RAZOR's capacitor expansion on the energy's charge dependence.

    Adds to the per-atom energy target

        E_i += (q / (eps0 A)) * p_z,i  +  [q phi_ref + q^2 / (2 C_0)] / N_real

    so that, summed over a structure's atoms,

        E(q) = E_0 + phi_0 q + q^2 / (2 C_0),
        phi_0 = phi_ref + P_z / (eps0 A),        P_z = sum_i p_z,i

    which is Eq. (1) of Bergmann, Reuter & Hoermann, J. Chem. Phys. 164,
    174110 (2026), with `phi_0` expressed through their Helmholtz relation,
    Eq. (4): P_z = eps0 A (phi_0 - phi_ref).

    Unlike `FiLMChargeScalar`, which leaves E(q) an arbitrary learned function,
    this makes E(q) *exactly* quadratic. Three consequences:

    - `dE/dq = phi_0 + q / C_0` and `d2E/dq2 = 1/C_0` exactly, so the curvature
      is a parameter rather than a by-product, and E(q) extrapolates in charge
      by construction rather than by whatever an MLP does off-distribution.
    - `dF_I/dq = -d(P_z)/dR_I` is independent of q, i.e. the SEBEC is the
      gradient of a learned polarization. This is the same structure as
      Z*_I = V dP/dR_I in the modern theory of polarization; the `eps0 A`
      cancels out of the response entirely.
    - The head is exactly padding-invariant (see PADDING below).

    WHY P_z RATHER THAN phi_0 DIRECTLY
    ----------------------------------
    `phi_0` is intensive, so it cannot be written as a plain sum over atoms; a
    mean-pooled `phi_0` would instead make every SEBEC scale as 1/N, which is
    wrong for a per-atom response of order e. `P_z` is a dipole: extensive,
    short-ranged, and the natural thing for an MLIP readout to sum. Intensivity
    of `phi_0` is then *derived*, through the 1/A. Adding slab thickness adds
    atoms with p_i ~ 0; doubling the area doubles P_z and A together.

    EQUIVARIANCE
    ------------
    `origin` must be l=1 with odd parity (`FunctionReduceParticular(selected_l=1,
    selected_p=-1)`), i.e. a true vector, not a pseudovector. This is not
    decoration: under a global mirror z -> -z every invariant descriptor is
    unchanged but P_z must flip sign, so an invariant readout could only
    memorise the fixed slab orientation of its training set.

    Parameters
    ----------
    origin : list[FunctionReduceParticular]
        l=1 instructions producing `[n_atoms, n_out, 3]`. Their contributions
        are summed, so a linear reduce and a gated one can be passed together
        to get the same linear/non-linear split `LinMLPOut2ScalarTarget` uses
        for the energy.
    area : float
        Surface area in Angstrom^2, used for the `1/(eps0 A)` conversion of
        `P_z` to a potential.

        **A fixed float, deliberately, not a tensor read from the cell.** The
        LAMMPS pair style (`src/ML-PACE/pair_grace.cpp`) feeds no cell or box
        input at all, so a model that requested `CELL_VECTORS` could not be run
        there. razor's cell is fixed at 82.31 A^2, so nothing is lost today.

        TODO: make the area part of the E(q) expression and obtain it at
        inference, so the model conditions on the surface charge density
        sigma = q/A instead of q and transfers across cell sizes. Two changes:
        this term's `1/(eps0 A)` becomes tensor-valued, and `inv_capacitance`
        (V/e, cell-specific) becomes an intensive `c_inv` in V A^2 / e with the
        capacitive energy `q^2 c_inv / (2 A)`. The blocker is on the LAMMPS
        side -- `pair_grace.cpp` needs to feed `cell_vectors` from `domain`
        before the instruction can request it. `FiLMChargeScalar`'s
        `normalize="per_area"` branch already has the `cross(cell[i], cell[j])`
        code to reuse. See notes/interim-report.md section 6 item 6 in
        lorem-q-work, which records this as a project-level intention.
    slab_normal_axis : int
        Cartesian component of the dipole that couples to the charge. **Explicit,
        never inferred from the cell** -- inferring it is how a silent 2.54x
        error in Born effective charges arose in a sibling project; see the
        same argument in `FiLMChargeScalar`.
    phi_ref : float
        Initial value of the reference work function in V, the `phi_0^ref` of
        Eq. (4). 5.75 V is the paper's clean Pt(111)-in-SJM value.

        This is a *conditioning device, not capacity*: it is exactly degenerate
        with a constant in `sum_i p_z,i`, so training it only lets it absorb the
        dataset mean. What it buys is the starting point -- with a
        zero-initialised dipole readout the model begins at
        `phi = phi_ref + q/C_0` and has only the small residual dipole left to
        learn, instead of having to build ~5 V out of a sum of atomic
        contributions. Same role `TrainableShiftTarget` plays for the energy.
    inv_capacitance : float
        Initial value of `1/C_0` in V/e. Unlike `phi_ref` this is *not*
        degenerate: it is what the `d2E/dq2` label pins directly. 9.16 V/e is
        the mean over `razor_centre.xyz` restricted to `polarizable=True`
        (+- 1.02 over 5398 frames, i.e. genuinely near structure-independent,
        which is what makes a single scalar defensible). Outside that window
        the label is a different number, not just a noisier one.

        Kept structure-independent on purpose. RAZOR assumes the same ("thus
        having no effect on forces"), and experiments/razor_quad/ in
        lorem-q-work records that a geometry-dependent per-atom curvature
        collapses to a constant and explains none of the variation, while a
        single global one had the best forces of that family.

    INITIALISATION
    --------------
    `FunctionReduceParticular` initialises its coefficients from a unit normal,
    so an untouched dipole readout emits a large *random* `P_z` -- measured at
    5.7 V RMS on the work function, against a label spread of ~1.2 V. That would
    swamp `phi_ref` and make its careful initial value pointless.

    So `dipole_scale` multiplies `P_z` and is initialised to **zero**, the same
    trick `FiLMChargeScalar.gate` and `InvariantLayerRMSNorm(init="zeros")` use.
    At step 0 the model is therefore exactly the bare capacitor,
    `phi = phi_ref + q / C_0`, which is the prior `phi_ref` exists to express.
    `dipole_scale` picks up a gradient immediately (dL/dscale is proportional to
    P_z, which is nonzero), and the reduce coefficients start moving as soon as
    it leaves zero.

    PADDING
    -------
    `p_z` is already exactly zero on a padded atom: its descriptors are zero,
    the reduce is linear without bias, and a gate multiplies rather than shifts.
    The charge-only bracket is not, so it is masked on `N_ATOMS_BATCH_REAL` and
    divided by the number of *real* atoms in the structure. Exactly `N_real`
    atoms then contribute `1/N_real` of it, so the structure total is exactly
    `q phi_ref + q^2 / (2 C_0)` at any padding width.

    That makes this head padding-exact, which FiLM is not -- see the note in
    `FiLMChargeScalar.frwrd` and the `padding 0` advice in LAMMPS'
    doc/src/pair_grace.rst. Do not rely on it without the test; verify it.
    """

    def __init__(
        self,
        origin: list,
        target: CreateOutputTarget,
        area: float,
        slab_normal_axis: int = 2,
        phi_ref: float = 5.75,
        inv_capacitance: float = 9.16,
        train_phi_ref: bool = True,
        train_inv_capacitance: bool = True,
        name: str = "HelmholtzChargeTarget",
        l: int = 0,  # noqa: E741
        **kwargs,
    ):
        super().__init__(name=name, target=target, l=l)

        assert slab_normal_axis in (0, 1, 2), "slab_normal_axis must be 0, 1 or 2"
        assert area > 0, f"area must be positive, got {area}"
        lmax = max(ins.lmax for ins in origin)
        assert lmax == 1, (
            f"HelmholtzChargeTarget needs l=1 (vector) origins, got lmax={lmax}. "
            f"Use FunctionReduceParticular(selected_l=1, selected_p=-1)."
        )
        self.assert_l_compatibility(target)

        self.origin = origin
        self.area = area
        self.slab_normal_axis = slab_normal_axis
        self.phi_ref_init = phi_ref
        self.inv_capacitance_init = inv_capacitance
        self.train_phi_ref = train_phi_ref
        self.train_inv_capacitance = train_inv_capacitance

        # Instance attribute rather than class attribute, so the requested keys
        # stay as narrow as possible -- the pattern FiLMChargeScalar uses. All
        # three are already fed by the LAMMPS pair style; `CELL_VECTORS`
        # deliberately is not requested (see `area` above).
        self.input_tensor_spec = {
            constants.TOTAL_CHARGE: {"shape": [None, 1], "dtype": "float"},
            constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
            constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        }

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if self.is_built:
            return
        self.phi_ref = tf.Variable(
            tf.constant(self.phi_ref_init, dtype=float_dtype),
            trainable=self.train_phi_ref,
            name="phi_ref",
        )
        self.inv_capacitance = tf.Variable(
            tf.constant(self.inv_capacitance_init, dtype=float_dtype),
            trainable=self.train_inv_capacitance,
            name="inv_capacitance",
        )
        # zero-init => P_z is exactly 0 at step 0, so the model starts as the
        # bare capacitor phi_ref + q/C_0 (see INITIALISATION above)
        self.dipole_scale = tf.Variable(
            tf.zeros([], dtype=float_dtype), name="dipole_scale"
        )
        self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        target = input_data[f"{self.target.name}"]

        p = 0.0
        for ins in self.origin:
            x = input_data[f"{ins.name}"]
            if getattr(ins, "lm_first", False):
                # [lm, atoms, n_out] -> [atoms, n_out, lm]
                x = tf.transpose(x, [1, 2, 0])
            # real spherical harmonics run m = -1, 0, +1, i.e. (y, z, x); the
            # roll is the same (y,z,x) -> (x,y,z) reorder LinearOut2EquivarTarget
            # applies, so slab_normal_axis indexes Cartesian components.
            p += tf.reduce_sum(tf.roll(x, shift=1, axis=2), axis=1)
        pz = p[:, self.slab_normal_axis : self.slab_normal_axis + 1]
        pz = pz * tf.cast(self.dipole_scale, pz.dtype)

        q = tf.cast(input_data[constants.TOTAL_CHARGE], pz.dtype)
        map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP]
        # each structure's charge, broadcast to its atoms. Padded atoms gather a
        # real structure's charge (LAMMPS maps every atom to structure 0), which
        # is why the bracket below has to be masked.
        q_at = tf.gather(q, map_at2struc)

        real = (
            tf.reshape(tf.range(tf.shape(pz)[0], dtype=tf.int32), [-1, 1])
            < input_data[constants.N_ATOMS_BATCH_REAL]
        )
        ones = tf.where(real, tf.ones_like(q_at), tf.zeros_like(q_at))
        n_real = tf.math.unsorted_segment_sum(
            ones, map_at2struc, num_segments=tf.shape(q)[0]
        )
        n_real_at = tf.gather(n_real, map_at2struc)

        eps0_area = tf.constant(cc.EPSILON_0, dtype=pz.dtype) * tf.constant(
            self.area, dtype=pz.dtype
        )
        phi_ref = tf.cast(self.phi_ref, pz.dtype)
        inv_c = tf.cast(self.inv_capacitance, pz.dtype)

        # geometry-dependent: q * P_z / (eps0 A). Already exactly zero on padded
        # atoms -- their descriptors are zero and the reduce is linear without a
        # bias -- so this term needs no mask of its own.
        dipole_term = q_at / eps0_area * pz
        # charge-only: q phi_ref + q^2 / (2 C_0), spread over the REAL atoms of
        # each structure so the structure total is exact at any padding width
        const_term = q_at * phi_ref + 0.5 * inv_c * tf.square(q_at)
        const_term = tf.math.divide_no_nan(const_term, n_real_at)
        const_term = tf.where(real, const_term, tf.zeros_like(const_term))

        return target + dipole_term + const_term
