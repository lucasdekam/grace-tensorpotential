"""Charge-conditioned GRACE: FiLM presets and the compute functions that go
with them.

Two presets, `GRACE_1LAYER_FILM` and `GRACE_2LAYER_FILM`, are verbatim copies of
`GRACE_1LAYER_v2_25` / `GRACE_2LAYER_v2_25` (tensorpotential/potentials/presets.py)
with `FiLMChargeScalar` inserted on every scalar tensor feeding the energy
readout. They are copies rather than wrappers so the shipped presets stay
untouched and a reviewer can diff the two side by side.

Three compute functions add `work_function = dE/dq`. Forces come from
d/d(bond_vector) and dE/dq from d/d(total_charge); a single non-persistent tape
produces both in one reverse sweep by passing a *list* of sources, so the work
function costs essentially nothing on top of the forces.

The module is named `model.py` so `extra/presets.py::load_extra_models` picks it
up, matching `extra/gen_tensor/model.py`.
"""

from __future__ import annotations

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.instructions import FiLMChargeScalar
from tensorpotential.instructions import (
    BondLength,
    BondSpecificRadialBasisFunction,
    ConstantScaleShiftTarget,
    CreateOutputTarget,
    FCRight2Left,
    FunctionReduceN,
    InstructionManager,
    InvariantLayerRMSNorm,
    LinMLPOut2ScalarTarget,
    LinearOut2Target,
    MLPRadialFunction_v2,
    ProductFunction,
    RadialBasis,
    ScalarChemicalEmbedding,
    ScaledBondVector,
    SingleParticleBasisFunctionEquivariantInd,
    SingleParticleBasisFunctionScalarInd,
    SphericalHarmonic,
    TrainableShiftTarget,
    ZBLPotential,
)
from tensorpotential.instructions.base import TPInstruction
from tensorpotential.potentials.registry import register_preset
from tensorpotential.tpmodel import (
    ComputeFunction,
    TrainFunction,
    compute_batch_virials_from_pair_forces,
    compute_structure_virials_from_pair_forces,
    execute_instructions,
)
from tensorpotential.utils import Parity


# ----------------------------------------------------------------------------
# presets
# ----------------------------------------------------------------------------


@register_preset(
    "GRACE_1LAYER_FILM",
    public=True,
    settings={
        "small": {
            "rcut": 6,
            "lmax": 4,
            "max_order": 4,
            "n_rad_max": 24,
            "prod_func_n_max": 48,
            "n_mlp_dens": 12,
            "n_rad_base": 8,
        },
        "medium": {
            "rcut": 6,
            "lmax": 4,
            "max_order": 4,
            "n_rad_max": 32,
            "prod_func_n_max": 64,
            "n_mlp_dens": 16,
            "n_rad_base": 10,
        },
        "large": {
            "rcut": 6,
            "lmax": 4,
            "max_order": 4,
            "n_rad_max": 42,
            "prod_func_n_max": 64,
            "n_mlp_dens": 16,
            "n_rad_base": 10,
        },
    },
)
def GRACE_1LAYER_FILM(
    element_map: dict,
    rcut: float = 6,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    lmax=4,
    basis_type: str = "Cheb",
    cutoff_function_order: int = 16,
    n_rad_base=10,
    n_rad_max=32,
    prod_func_n_max=64,
    embedding_size=128,
    n_mlp_dens: int = 16,
    max_order: int = 4,
    func_init="random",
    chem_init="random",
    cutoff_dict: dict = None,
    atomic_shift_map: dict = None,
    zbl_cutoff: dict = None,
    dense_nbr: bool = False,
    charge_hidden_layers: list = None,
    charge_activation: str = "silu",
    charge_normalize: str = "none",
    slab_normal_axis: int = 2,
    modulate_linear_channel: bool = True,
    **kwargs,
) -> InstructionManager:
    """`GRACE_1LAYER_v2_25` with FiLM charge conditioning on `rho`.

    `rho` is the single scalar node-feature tensor feeding the readout, so one
    FiLM instruction covers the whole energy.
    """
    num_elements = len(element_map)
    film_kwargs = dict(
        hidden_layers=charge_hidden_layers,
        activation=charge_activation,
        normalize=charge_normalize,
        slab_normal_axis=slab_normal_axis,
        modulate_linear_channel=modulate_linear_channel,
    )
    with InstructionManager(dense_nbr=dense_nbr) as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        if cutoff_dict is None:
            g_k = RadialBasis(
                bonds=d_ij,
                basis_type=basis_type,
                nfunc=n_rad_base,
                p=cutoff_function_order,
                normalized=False,
                rcut=rcut,
            )
        else:
            g_k = BondSpecificRadialBasisFunction(
                bonds=d_ij,
                element_map=element_map,
                cutoff_dict=cutoff_dict,
                cutoff=rcut,
                cutoff_type="symmetric_bond",
                cutoff_function_param=cutoff_function_order,
                basis_type=basis_type,
                nfunc=n_rad_base,
            )

        z = ScalarChemicalEmbedding(
            element_map=element_map,
            embedding_size=embedding_size,
            name="Z",
            init=chem_init,
        )

        R_nl = MLPRadialFunction_v2(
            n_rad_max=n_rad_max,
            lmax=lmax,
            basis=g_k,
            name="R",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=avg_n_neigh
        )

        instructions = [A]

        if max_order > 1:
            A1 = FCRight2Left(
                left=A, right=A, name="A1", n_out=prod_func_n_max, norm_out=True
            )
            AA = ProductFunction(
                left=A1,
                right=A1,
                name="AA",
                lmax=lmax,
                Lmax=lmax,
                keep_parity=Parity.REAL_PARITY,
                is_left_right_equal=True,
                normalize=True,
            )
            instructions.append(AA)

        if max_order > 2:
            AA1 = FCRight2Left(
                left=AA, right=A, name="AA1", n_out=prod_func_n_max, norm_out=True
            )
            AAA = ProductFunction(
                left=AA1,
                right=A1,
                name="AAA",
                lmax=lmax,
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAA)
        if max_order > 3:
            AA2 = FCRight2Left(
                left=AA, right=A, name="AA2", n_out=prod_func_n_max, norm_out=True
            )
            AAAA = ProductFunction(
                left=AA2,
                right=AA2,
                name="AAAA",
                lmax=lmax,
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)

        instr_red = FunctionReduceN(
            instructions=instructions,
            name="rho",
            ls_max=0,
            n_out=n_mlp_dens + 1,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.REAL_PARITY,
            init_vars=func_init,
        )

        # >>> the only departure from GRACE_1LAYER_v2_25 <<<
        rho_film = FiLMChargeScalar(inpt=instr_red, name="rho_film", **film_kwargs)

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinMLPOut2ScalarTarget(
            origin=[rho_film], target=out_instr, hidden_layers=[64], activation="silu"
        )
        if (
            (constant_out_shift != 0)
            or (constant_out_scale != 1)
            or (atomic_shift_map is not None)
        ):
            ConstantScaleShiftTarget(
                target=out_instr,
                scale=constant_out_scale,
                shift=constant_out_shift,
                atomic_shift_map=atomic_shift_map,
            )
        TrainableShiftTarget(target=out_instr, number_of_atom_types=num_elements)
        if zbl_cutoff is not None:
            zbl = ZBLPotential(bonds=d_ij, cutoff=zbl_cutoff, element_map=element_map)
            LinearOut2Target(origin=[zbl], target=out_instr, name="zbl_output")

    return instructor


@register_preset(
    "GRACE_2LAYER_FILM",
    public=True,
    settings={
        "small": {
            "rcut": 6,
            "lmax": [4, 3],
            "max_order": 4,
            "indicator_lmax": 1,
            "n_rad_max": [32, 32],
            "prod_func_n_max": [32, 32],
            "n_mlp_dens": 12,
            "n_rad_base": 8,
        },
        "medium": {
            "rcut": 6,
            "lmax": [4, 3],
            "max_order": 4,
            "indicator_lmax": 1,
            "n_rad_max": [42, 32],
            "prod_func_n_max": [42, 64],
            "n_mlp_dens": 16,
            "n_rad_base": 10,
        },
        "large": {
            "rcut": 6,
            "lmax": [4, 3],
            "max_order": 4,
            "indicator_lmax": 3,
            "n_rad_max": [42, 32],
            "prod_func_n_max": [42, 64],
            "n_mlp_dens": 16,
            "n_rad_base": 10,
        },
    },
)
def GRACE_2LAYER_FILM(
    element_map: dict,
    rcut: float = 6,
    avg_n_neigh: float = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    lmax=(4, 3),
    basis_type: str = "Cheb",
    cutoff_function_order: int = 16,
    n_rad_base=10,
    n_rad_max=(42, 32),
    prod_func_n_max=(42, 64),
    embedding_size=128,
    n_mlp_dens: int = 16,
    max_order: int = 4,
    indicator_lmax: int = 3,
    func_init="random",
    chem_init="random",
    cutoff_dict: dict = None,
    atomic_shift_map: dict = None,
    zbl_cutoff: dict = None,
    dense_nbr: bool = False,
    charge_hidden_layers: list = None,
    charge_activation: str = "silu",
    charge_normalize: str = "none",
    slab_normal_axis: int = 2,
    modulate_linear_channel: bool = True,
    **kwargs,
) -> InstructionManager:
    """`GRACE_2LAYER_v2_25` with FiLM charge conditioning on both readout branches.

    The readout sums `I_out_0_LN` (layer 1) and `I_1_LN` (layer 2), so BOTH are
    conditioned -- FiLMing only one would leave the other's energy contribution
    charge-independent.

    Note `I_out_0` is a readout branch, not the input to layer 2: layer 2
    consumes the *equivariant* indicator `I` via `YI`. So this is not "first
    layer vs last layer" conditioning; both FiLMs sit on the invariant readout
    path, and nothing in the message passing sees the charge.
    """
    if isinstance(lmax, int):
        lmax = [lmax, lmax]
    if isinstance(n_rad_max, int):
        n_rad_max = [n_rad_max, n_rad_max]
    if isinstance(prod_func_n_max, int):
        prod_func_n_max = [prod_func_n_max, prod_func_n_max]

    assert prod_func_n_max[0] == n_rad_max[0], (
        f"n_rad_max[0] and prod_func_n_max[0] must match, "
        f"but {n_rad_max[0]} and {prod_func_n_max[0]} instead."
    )
    Il = indicator_lmax
    num_elements = len(element_map)
    film_kwargs = dict(
        hidden_layers=charge_hidden_layers,
        activation=charge_activation,
        normalize=charge_normalize,
        slab_normal_axis=slab_normal_axis,
        modulate_linear_channel=modulate_linear_channel,
    )
    with InstructionManager(dense_nbr=dense_nbr) as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)

        if cutoff_dict is None:
            g_k = RadialBasis(
                bonds=d_ij,
                basis_type=basis_type,
                nfunc=n_rad_base,
                p=cutoff_function_order,
                normalized=False,
                rcut=rcut,
            )
        else:
            g_k = BondSpecificRadialBasisFunction(
                bonds=d_ij,
                element_map=element_map,
                cutoff_dict=cutoff_dict,
                cutoff=rcut,
                cutoff_type="symmetric_bond",
                cutoff_function_param=cutoff_function_order,
                basis_type=basis_type,
                nfunc=n_rad_base,
            )

        Y = SphericalHarmonic(vhat=rhat, lmax=lmax[0], name="Y")
        z = ScalarChemicalEmbedding(
            element_map=element_map,
            embedding_size=embedding_size,
            name="Z",
            init=chem_init,
        )

        R_nl = MLPRadialFunction_v2(
            n_rad_max=n_rad_max[0],
            lmax=lmax[0],
            basis=g_k,
            name="R",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )

        A = SingleParticleBasisFunctionScalarInd(
            radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=avg_n_neigh
        )

        instructions = [A]

        if max_order > 1:
            A1 = FCRight2Left(
                left=A, right=A, name="A1", n_out=prod_func_n_max[0], norm_out=True
            )
            AA = ProductFunction(
                left=A1,
                right=A1,
                name="AA",
                lmax=lmax[0],
                Lmax=lmax[0],
                keep_parity=Parity.REAL_PARITY,
                is_left_right_equal=True,
                normalize=True,
            )
            instructions.append(AA)

        if max_order > 2:
            AA1 = FCRight2Left(
                left=AA,
                right=A,
                name="AA1",
                n_out=prod_func_n_max[0],
                norm_out=True,
            )
            AAA = ProductFunction(
                left=AA1,
                right=A,
                name="AAA",
                lmax=lmax[0],
                Lmax=Il,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAA)
        if max_order > 3:
            AA2 = FCRight2Left(
                left=AA,
                right=A,
                name="AA2",
                n_out=prod_func_n_max[0],
                norm_out=True,
            )
            AAAA = ProductFunction(
                left=AA2,
                right=AA2,
                name="AAAA",
                lmax=lmax[0],
                Lmax=1 if Il > 0 else 0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions.append(AAAA)

        I1 = FunctionReduceN(
            name="I1",
            instructions=instructions,
            ls_max=[Il, Il, Il, 1 if Il > 0 else 0][: len(instructions)],
            n_out=12,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.REAL_PARITY,
        )

        instr_red = FunctionReduceN(
            name="I",
            instructions=[I1],
            ls_max=[Il],
            n_out=n_rad_max[1],
            is_central_atom_type_dependent=False,
            allowed_l_p=Parity.REAL_PARITY,
            init_vars=func_init,
        )

        I_0 = FunctionReduceN(
            instructions=instructions,
            name="I_out_0",
            ls_max=0,
            n_out=n_mlp_dens + 1,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
            init_vars=func_init,
        )
        I_0_LN = InvariantLayerRMSNorm(
            inpt=I_0,
            name="I_out_0_LN",
            type="only_nonlin",
        )

        R1_nl = MLPRadialFunction_v2(
            n_rad_max=n_rad_max[1],
            lmax=lmax[0],
            basis=g_k,
            name="R1",
            hidden_layers=[64, 64],
            activation=["silu", "silu"],
        )
        B0 = SingleParticleBasisFunctionScalarInd(
            radial=R1_nl,
            angular=Y,
            indicator=z,
            name="B0",
            avg_n_neigh=avg_n_neigh,
        )

        YI = SingleParticleBasisFunctionEquivariantInd(
            radial=R1_nl,
            angular=Y,
            indicator=instr_red,
            name="YI",
            lmax=lmax[0],
            Lmax=lmax[1],
            avg_n_neigh=avg_n_neigh,
            keep_parity=Parity.FULL_PARITY,
            normalize=True,
        )
        B = FunctionReduceN(
            instructions=[YI, B0],
            name="B",
            ls_max=lmax[1],
            out_norm=False,
            n_out=prod_func_n_max[1],
            is_central_atom_type_dependent=False,
            allowed_l_p=Parity.FULL_PARITY,
        )
        instructions2 = [B]

        if max_order > 1:
            B1 = FCRight2Left(
                left=B,
                right=B,
                name="B1",
                n_out=prod_func_n_max[1],
                norm_out=True,
            )
            BB = ProductFunction(
                left=B1,
                right=B1,
                name="BB",
                lmax=lmax[1],
                Lmax=lmax[1],
                keep_parity=Parity.FULL_PARITY + [[0, -1]],
                is_left_right_equal=True,
                normalize=True,
            )
            instructions2.append(BB)
        if max_order > 2:
            BB1 = FCRight2Left(
                left=BB,
                right=B,
                name="BB1",
                n_out=prod_func_n_max[1],
                norm_out=True,
            )
            BBB = ProductFunction(
                left=BB1,
                right=B,
                name="BBB",
                lmax=lmax[1],
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BBB)
        if max_order > 3:
            BB2 = FCRight2Left(
                left=BB,
                right=B,
                name="BB2",
                n_out=prod_func_n_max[1],
                norm_out=True,
            )
            BBBB = ProductFunction(
                left=BB2,
                right=BB2,
                name="BBBB",
                lmax=lmax[1],
                Lmax=0,
                keep_parity=Parity.REAL_PARITY,
                normalize=True,
            )
            instructions2.append(BBBB)

        I_1 = FunctionReduceN(
            instructions=instructions2,
            name="I_out_1",
            ls_max=0,
            n_out=n_mlp_dens + 1,
            is_central_atom_type_dependent=True,
            number_of_atom_types=num_elements,
            allowed_l_p=Parity.SCALAR,
            init_vars=func_init,
        )
        I_1_LN = InvariantLayerRMSNorm(
            inpt=I_1,
            name="I_1_LN",
            type="full",
        )

        # >>> the only departure from GRACE_2LAYER_v2_25 <<<
        I_0_film = FiLMChargeScalar(inpt=I_0_LN, name="I_out_0_film", **film_kwargs)
        I_1_film = FiLMChargeScalar(inpt=I_1_LN, name="I_1_film", **film_kwargs)

        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        LinMLPOut2ScalarTarget(
            origin=[I_0_film, I_1_film],
            target=out_instr,
            hidden_layers=[64],
            activation="tanh",
        )
        if (
            (constant_out_shift != 0)
            or (constant_out_scale != 1)
            or (atomic_shift_map is not None)
        ):
            ConstantScaleShiftTarget(
                target=out_instr,
                scale=constant_out_scale,
                shift=constant_out_shift,
                atomic_shift_map=atomic_shift_map,
            )
        TrainableShiftTarget(target=out_instr, number_of_atom_types=num_elements)
        if zbl_cutoff is not None:
            zbl = ZBLPotential(bonds=d_ij, cutoff=zbl_cutoff, element_map=element_map)
            LinearOut2Target(origin=[zbl], target=out_instr, name="zbl_output")

    # `I_out_0_LN` is still the layer-1 tensor communicated across the multi-GPU
    # graph split; the FiLM that consumes it lives in the second half.
    instructor.communicated_keys = ["I_out_0_LN", "I"]
    return instructor


# ----------------------------------------------------------------------------
# compute functions
# ----------------------------------------------------------------------------

_CHARGE_SPECS = {
    constants.TOTAL_CHARGE: {"shape": [None, 1], "dtype": "float"},
}


def _tape_energy_forces_charge(instructions, input_data, training, local=False):
    """One tape, one reverse sweep, both gradients.

    `tape.gradient` accepts a *list* of sources and differentiates them in a
    single backward pass, so dE/dq is essentially free on top of the forces.

    Returns `(e_atomic, pair_f, dE_dq)`. `dE_dq` is None when the graph does not
    depend on the charge, which happens if one of these compute functions is
    paired with a model that has no FiLM instruction.
    """
    q = input_data[constants.TOTAL_CHARGE]
    with tf.GradientTape() as tape:
        tape.watch(input_data[constants.BOND_VECTOR])
        tape.watch(q)
        execute_instructions(input_data, instructions, training, local=local)
        e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])
    g_bond, g_q = tape.gradient(e_atomic, [input_data[constants.BOND_VECTOR], q])

    pair_f = tf.negative(g_bond)
    e_atomic = tf.cast(e_atomic, dtype=pair_f.dtype)
    # Each structure's energy depends only on its own charge, so the gradient of
    # the summed energy w.r.t. the [n_struct, 1] charge vector is already the
    # per-structure dE/dq -- no Jacobian needed. Sign convention:
    # work_function = dE/dq, matching LOREM and the razor/natcomm2025 labels.
    if g_q is not None:
        # FiLM reaches the charge through tf.gather (broadcasting one charge to
        # a structure's atoms), and the gradient of a gather is *sparse*, so the
        # tape hands back IndexedSlices rather than a Tensor. Densify it, or
        # every downstream consumer -- loss, metrics, .numpy() -- breaks.
        if isinstance(g_q, tf.IndexedSlices):
            g_q = tf.convert_to_tensor(g_q)
        g_q = tf.cast(g_q, dtype=pair_f.dtype)
    return e_atomic, pair_f, g_q


def _batch_energy_forces(input_data, e_atomic, pair_f):
    total_energy = tf.math.unsorted_segment_sum(
        e_atomic,
        input_data[constants.ATOMS_TO_STRUCTURE_MAP],
        num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
    )
    nat = tf.reshape(input_data[constants.N_ATOMS_BATCH_TOTAL], [])
    total_f = tf.math.unsorted_segment_sum(
        pair_f, input_data[constants.BOND_IND_J], num_segments=nat
    ) - tf.math.unsorted_segment_sum(
        pair_f, input_data[constants.BOND_IND_I], num_segments=nat
    )
    return total_energy, total_f


class ComputeBatchEnergyForcesCharge(TrainFunction):
    """`ComputeBatchEnergyAndForces` plus `work_function = dE/dq`."""

    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        **_CHARGE_SPECS,
    }

    def __init__(self, extra_return_keys: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.extra_return_keys = extra_return_keys

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        e_atomic, pair_f, dedq = _tape_energy_forces_charge(
            instructions, input_data, training
        )
        total_energy, total_f = _batch_energy_forces(input_data, e_atomic, pair_f)
        res = {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            cc.PREDICT_WORK_FUNCTION: (
                tf.zeros_like(total_energy) if dedq is None else dedq
            ),
        }
        if self.extra_return_keys:
            for k in self.extra_return_keys:
                if k in input_data:
                    res[k] = input_data[k]
        return res


class ComputeBatchEnergyForcesVirialsCharge(TrainFunction):
    """`ComputeBatchEnergyForcesVirials` plus `work_function = dE/dq`."""

    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.BONDS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        **_CHARGE_SPECS,
    }

    def __init__(self, extra_return_keys: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.extra_return_keys = extra_return_keys

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        e_atomic, pair_f, dedq = _tape_energy_forces_charge(
            instructions, input_data, training
        )
        total_energy, total_f = _batch_energy_forces(input_data, e_atomic, pair_f)
        res = {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            constants.PREDICT_VIRIAL: compute_batch_virials_from_pair_forces(
                pair_f, input_data
            ),
            cc.PREDICT_WORK_FUNCTION: (
                tf.zeros_like(total_energy) if dedq is None else dedq
            ),
        }
        if self.extra_return_keys:
            for k in self.extra_return_keys:
                if k in input_data:
                    res[k] = input_data[k]
        return res


class ComputeStructureEnergyForcesVirialCharge(ComputeFunction):
    """Single-structure inference variant, for the ASE calculator and export.

    Mirrors `ComputeStructureEnergyAndForcesAndVirial`: energy is a plain
    reduce_sum (one structure), `nat` comes from `ATOMIC_MU_I`, and the virial
    uses the structure-level helper. The `"z_"` prefix on the pair forces is
    kept from the original -- SavedModel outputs are sorted alphabetically and
    the prefix pins it last.
    """

    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        **_CHARGE_SPECS,
    }

    def __init__(self, local=False, extra_return_keys: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.local = local
        self.extra_return_keys = extra_return_keys
        if self.local:
            self.specs[constants.ATOMIC_MU_I_LOCAL] = {"shape": [None], "dtype": "int"}

    def __call__(
        self,
        instructions: dict | list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        e_atomic, pair_f, dedq = _tape_energy_forces_charge(
            instructions, input_data, training, local=self.local
        )
        # Mask padded atoms before summing. The stock
        # ComputeStructureEnergyAndForcesAndVirial reduce_sums every entry, and
        # the ASE calculator appends fake atoms whose atomic energy is NOT zero
        # -- an isolated atom still gets a chemical embedding and an
        # element-dependent reduce, so it carries that species' isolated-atom
        # energy. Measured at a constant -0.1174 eV (one padded atom) on
        # natcomm2025, i.e. 0.504 meV/atom of pure bias.
        #
        # Harmless for forces (a constant) and for MD at fixed padding, but the
        # bias scales with the NUMBER of padded atoms, which adaptive padding
        # varies per structure -- so energy *differences* between differently
        # sized structures are wrong too. Other instructions here already mask
        # on N_ATOMS_BATCH_REAL (InvariantLayerRMSNorm, TrainableShiftTarget);
        # this sum did not.
        n_at_total = tf.shape(e_atomic)[0]
        n_at_real = input_data[constants.N_ATOMS_BATCH_REAL]
        real = tf.reshape(tf.range(n_at_total, dtype=tf.int32), [-1, 1]) < n_at_real
        total_energy = tf.reduce_sum(
            tf.where(real, e_atomic, tf.zeros_like(e_atomic)), axis=0, keepdims=True
        )
        nat = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        total_f = tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_I], num_segments=nat
        )
        res = {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_VIRIAL: compute_structure_virials_from_pair_forces(
                pair_f, input_data
            ),
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            cc.PREDICT_WORK_FUNCTION: (
                tf.zeros_like(total_energy) if dedq is None else dedq
            ),
            "z_" + constants.PREDICT_PAIR_FORCES: pair_f,
        }
        if self.extra_return_keys:
            for k in self.extra_return_keys:
                if k in input_data:
                    res[k] = input_data[k]
        return res
