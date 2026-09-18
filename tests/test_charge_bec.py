"""Born effective charges and d2E/dq2 from a charge-conditioned GRACE model.

`predict_bec=True` adds a second reverse sweep, over the *scalar* dE/dq:

    dF/dq = -d2E/(dr dq) = -d/dr (dE/dq)

so the second derivative is an ordinary gradient of the same shape the forces
already take, not a Jacobian. Forward mode would be the textbook choice for a
[n_atoms, 3] output against one scalar per structure, but
`tf.autodiff.ForwardAccumulator` cannot handle the `IndexedSlices` that FiLM's
`tf.gather` produces -- it raises `'IndexedSlices' object has no attribute
'_id'` inside the inner tape.

These tests pin both derivatives against central differences of the quantities
the model itself reports, and pin the label conversion in the data builder.
"""

import numpy as np
import pytest
import tensorflow as tf
from ase.build import bulk

from tensorpotential import constants
from tensorpotential.calculator import TPCalculator
from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.databuilder import TotalChargeDataBuilder
from tensorpotential.extra.charge.model import ComputeStructureEnergyForcesVirialCharge
from tensorpotential.potentials import get_preset
from tensorpotential.tpmodel import TPModel

ELEMENT_MAP = {"Al": 0}
SMALL = dict(rcut=5.0, n_mlp_dens=4, embedding_size=16, max_order=2, n_rad_base=6)
PRESETS = ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"]


def _instructions(preset, **kwargs):
    if "2LAYER" in preset:
        kwargs.setdefault("lmax", [2, 2])
        kwargs.setdefault("n_rad_max", [8, 8])
        kwargs.setdefault("prod_func_n_max", [8, 8])
        kwargs.setdefault("indicator_lmax", 1)
    else:
        kwargs.setdefault("lmax", 2)
        kwargs.setdefault("n_rad_max", 8)
        kwargs.setdefault("prod_func_n_max", 8)
    return get_preset(preset)(element_map=ELEMENT_MAP, **{**SMALL, **kwargs}).get_instructions()


def _model(preset, predict_bec=True):
    # graph-level seed, not just the op-level one: tf.random.normal(seed=...)
    # alone still depends on how many random ops ran before it, so the gate
    # values -- and with them the size of the derivatives under test -- would
    # otherwise change with test ordering.
    tf.random.set_seed(20260918)
    model = TPModel(
        _instructions(preset),
        compute_function=ComputeStructureEnergyForcesVirialCharge(
            predict_bec=predict_bec
        ),
    )
    model.build(tf.float64)
    n = 0
    for v in model.variables:
        if "gate" in v.name:
            v.assign(tf.random.normal(v.shape, stddev=0.5, dtype=v.dtype, seed=3))
            n += 1
    assert n > 0, "no FiLM gate variable found"
    return model


def _atoms():
    at = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 1, 1)
    at.rattle(0.05, seed=1)
    return at


def _run(calc, at, q):
    """energy, forces, dE/dq, dF/dq, d2E/dq2 at charge `q`."""
    a = at.copy()
    a.info[cc.INFO_TOTAL_CHARGE] = float(q)
    a.calc = calc
    calc.reset()  # atoms.info is invisible to ASE's check_state
    e = a.get_potential_energy()
    f = a.get_forces()
    out = calc.outputs[0]
    wf = float(np.asarray(out[cc.PREDICT_WORK_FUNCTION]).ravel()[0])
    if cc.PREDICT_DF_DQ not in out:  # predict_bec=False
        return e, f, wf, None, None
    # the raw output keeps the padded atom; ASE's forces do not
    dfdq = np.asarray(out[cc.PREDICT_DF_DQ])[: len(a)]
    d2 = float(np.asarray(out[cc.PREDICT_D2E_DQ2]).ravel()[0])
    return e, f, wf, dfdq, d2


@pytest.mark.parametrize("preset", PRESETS)
def test_dfdq_matches_central_difference_of_forces(preset):
    """dF/dq is the charge derivative of the forces the same model reports."""
    calc = TPCalculator(model=_model(preset))
    at = _atoms()

    q, h = 0.4, 1e-4
    dfdq = _run(calc, at, q)[3]
    fd = (_run(calc, at, q + h)[1] - _run(calc, at, q - h)[1]) / (2 * h)

    scale = np.abs(fd).max()
    assert scale > 1e-3, "gate closed: nothing to test"
    # the reference is a central difference, so the comparison is limited by
    # its O(h^2) truncation error rather than by autodiff
    np.testing.assert_allclose(dfdq, fd, rtol=1e-4, atol=1e-6 * max(scale, 1.0))


@pytest.mark.parametrize("preset", PRESETS)
def test_d2edq2_matches_central_difference_of_the_work_function(preset):
    """It rides on the same sweep as dF/dq, so it is checked the same way."""
    calc = TPCalculator(model=_model(preset))
    at = _atoms()

    q, h = 0.4, 1e-4
    d2 = _run(calc, at, q)[4]
    fd = (_run(calc, at, q + h)[2] - _run(calc, at, q - h)[2]) / (2 * h)

    assert abs(fd) > 1e-3, "gate closed: nothing to test"
    assert d2 == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize("preset", PRESETS)
def test_energy_and_forces_are_unchanged_by_predict_bec(preset):
    """The extra sweep is an extra *output*, not a different model."""
    at = _atoms()
    ref = _model(preset, predict_bec=False)
    with_bec = TPModel(
        _instructions(preset),
        compute_function=ComputeStructureEnergyForcesVirialCharge(predict_bec=True),
    )
    with_bec.build(tf.float64)
    # same weights, so any difference is the compute function's doing
    for a, b in zip(ref.variables, with_bec.variables):
        b.assign(a)

    e0, f0, wf0 = _run(TPCalculator(model=ref), at, 0.7)[:3]
    e1, f1, wf1 = _run(TPCalculator(model=with_bec), at, 0.7)[:3]

    assert e1 == pytest.approx(e0, rel=1e-12)
    np.testing.assert_allclose(f1, f0, rtol=1e-10, atol=1e-12)
    assert wf1 == pytest.approx(wf0, rel=1e-10)


def test_bec_label_is_converted_once_and_explicitly():
    """Z* = (A eps0) dF/dq, with the area taken perpendicular to the named
    axis. A slab normal along `c` is spanned by `a` and `b`; getting that wrong
    is a silent rescaling of the whole target."""
    at = _atoms()
    z = np.arange(3 * len(at), dtype=float).reshape(len(at), 3)
    at.arrays[cc.ARRAYS_BEC_Z] = z
    at.info[cc.INFO_TOTAL_CHARGE] = 0.25

    builder = TotalChargeDataBuilder(fit_df_dq=True, bec_normal_axis=2)
    got = builder.extract_from_ase_atoms(at)[cc.DATA_REFERENCE_DF_DQ]

    cell = np.asarray(at.get_cell())
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    np.testing.assert_allclose(got, z / (area * cc.EPSILON_0), rtol=1e-12)

    # None means the label is already raw dF/dq and nothing is applied
    raw = TotalChargeDataBuilder(fit_df_dq=True, bec_normal_axis=None)
    np.testing.assert_allclose(
        raw.extract_from_ase_atoms(at)[cc.DATA_REFERENCE_DF_DQ], z, rtol=1e-12
    )


def test_missing_bec_label_is_an_error_not_a_zero():
    """A silently absent target trains on nothing and looks healthy."""
    at = _atoms()
    at.info[cc.INFO_TOTAL_CHARGE] = 0.0
    builder = TotalChargeDataBuilder(fit_df_dq=True)
    with pytest.raises(KeyError, match=cc.ARRAYS_BEC_Z):
        builder.extract_from_ase_atoms(at)


# -- the training path ------------------------------------------------------
#
# The tests above drive `ComputeStructureEnergyForcesVirialCharge`, which is
# what the ASE calculator and the export use. Training goes through
# `ComputeBatchEnergyForcesCharge` instead, with different plumbing: real
# batches, padded atoms and structures, and the segment maps. These pin that
# path against the single-structure one.


def test_batched_dfdq_matches_the_single_structure_path():
    """Same weights, same structure, two compute functions -- one batched and
    padded, one not. Padding is where a per-atom label goes wrong quietly."""
    from tensorpotential.data.databuilder import (
        construct_batches,
        GeometricalDataBuilder,
    )
    from tensorpotential.extra.charge.model import ComputeBatchEnergyForcesCharge

    at = _atoms()
    rng = np.random.default_rng(0)
    at.arrays[cc.ARRAYS_BEC_Z] = rng.normal(size=(len(at), 3))
    at.info[cc.INFO_TOTAL_CHARGE] = 0.4

    instr = _instructions("GRACE_1LAYER_FILM")
    batch_model = TPModel(
        instr, compute_function=ComputeBatchEnergyForcesCharge(predict_bec=True)
    )
    batch_model.build(tf.float64)
    tf.random.set_seed(20260918)
    for v in batch_model.variables:
        if "gate" in v.name:
            v.assign(tf.random.normal(v.shape, stddev=0.5, dtype=v.dtype, seed=3))

    single = TPModel(
        instr,
        compute_function=ComputeStructureEnergyForcesVirialCharge(predict_bec=True),
    )
    single.build(tf.float64)
    for a, b in zip(batch_model.variables, single.variables):
        b.assign(a)

    batches, _ = construct_batches(
        [at, at],
        data_builders=[
            GeometricalDataBuilder(ELEMENT_MAP, cutoff=5.0),
            TotalChargeDataBuilder(fit_df_dq=True, fit_d2e_dq2=False,
                                   normalize_weights=False),
        ],
        batch_size=2,
        max_n_buckets=1,
        return_padding_stats=True,
        verbose=False,
    )
    # construct_batches hands back numpy; the tapes need tensors to watch
    batch = {
        k: (v if isinstance(v, tf.Tensor) else tf.constant(v))
        for k, v in batches[0].items()
    }
    out = batch_model.compute(batch)

    n_real = int(np.asarray(batches[0][constants.N_ATOMS_BATCH_REAL]))
    got = np.asarray(out[cc.PREDICT_DF_DQ])[:n_real]
    ref = _run(TPCalculator(model=single), at, 0.4)[3]

    # two identical structures in the batch, so both halves equal the single
    assert n_real == 2 * len(at)
    np.testing.assert_allclose(got[: len(at)], ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got[len(at):], ref, rtol=1e-8, atol=1e-10)

    # the label survives batching with its components intact, and the padded
    # atoms carry no weight
    label = np.asarray(batches[0][cc.DATA_REFERENCE_DF_DQ])
    weight = np.asarray(batches[0][cc.DATA_DF_DQ_WEIGHTS])
    cell = np.asarray(at.get_cell())
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    expected = at.arrays[cc.ARRAYS_BEC_Z] / (area * cc.EPSILON_0)
    np.testing.assert_allclose(label[: len(at)], expected, rtol=1e-12)
    assert label.shape[0] == weight.shape[0] >= n_real
    assert np.all(weight[n_real:] == 0.0), "padded atoms must carry zero weight"
    assert np.all(weight[:n_real] == 1.0)


def test_predict_bec_can_be_switched_on_from_input_yaml():
    """gracemaker builds a compute function as `Cls(compute_function_config=...)`
    and forwards nothing else, so that dict is the only route from the YAML."""
    from tensorpotential.extra.charge.model import ComputeBatchEnergyForcesCharge

    assert not ComputeBatchEnergyForcesCharge().predict_bec
    assert not ComputeBatchEnergyForcesCharge(compute_function_config={}).predict_bec
    assert ComputeBatchEnergyForcesCharge(
        compute_function_config={"predict_bec": True}
    ).predict_bec
    # the explicit keyword still works, for the tests and the ASE calculator
    assert ComputeStructureEnergyForcesVirialCharge(predict_bec=True).predict_bec
    assert ComputeStructureEnergyForcesVirialCharge(
        compute_function_config={"predict_bec": True}
    ).predict_bec
