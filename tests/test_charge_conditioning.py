"""Tests for FiLM charge conditioning and the dE/dq work function.

The backwards-compatibility tests are the important ones: an existing GRACE
model must keep working with no charge specified anywhere.

Note ASE's `check_state` compares positions/numbers/cell/pbc but NOT
`atoms.info`, so changing only the charge does not invalidate the calculator's
cache. Every evaluation here goes through `_run`, which calls `calc.reset()`.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pytest
import tensorflow as tf
from ase.build import bulk

from tensorpotential import constants
from tensorpotential.calculator import TPCalculator
from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.model import ComputeStructureEnergyForcesVirialCharge
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.tpmodel import TPModel, ComputeStructureEnergyAndForcesAndVirial

ELEMENT_MAP = {"Al": 0}
# deliberately tiny -- these tests are about plumbing and derivatives, not accuracy
SMALL = dict(
    rcut=5.0, n_mlp_dens=4, embedding_size=16, max_order=2, n_rad_base=6
)


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
    return get_preset(preset)(
        element_map=ELEMENT_MAP, **{**SMALL, **kwargs}
    ).get_instructions()


def _model(preset, charge_aware=True, **kwargs):
    compute_fn = (
        ComputeStructureEnergyForcesVirialCharge()
        if charge_aware
        else ComputeStructureEnergyAndForcesAndVirial()
    )
    model = TPModel(_instructions(preset, **kwargs), compute_function=compute_fn)
    model.build(tf.float64)
    return model


def _atoms():
    at = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 1, 1)
    at.rattle(0.05, seed=1)
    return at


def _run(calc, at, q, positions=None):
    """Energy, forces and dE/dq at charge `q`."""
    a = at.copy()
    if positions is not None:
        a.set_positions(positions)
    a.info[cc.INFO_TOTAL_CHARGE] = float(q)
    a.calc = calc
    calc.reset()  # atoms.info is invisible to ASE's check_state
    energy = a.get_potential_energy()
    forces = a.get_forces()
    wf = calc.outputs[0][cc.PREDICT_WORK_FUNCTION].numpy().ravel()[0]
    return energy, forces, wf


def _set_gate(model, value):
    """FiLM is gated to zero at init; open it to exercise the conditioning."""
    n = 0
    for v in model.variables:
        if "gate" in v.name:
            v.assign(
                tf.zeros_like(v)
                if value is None
                else tf.random.normal(v.shape, stddev=value, dtype=v.dtype, seed=3)
            )
            n += 1
    assert n > 0, "no FiLM gate variable found"
    return n


# -- backwards compatibility -------------------------------------------------


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_latest", "GRACE_2LAYER_latest"])
def test_unconditioned_model_never_requires_a_charge(preset):
    """The whole backwards-compatibility guarantee, in one assertion.

    `total_charge` enters the tf.function signature only through a FiLM
    instruction's `input_tensor_spec`, so a stock preset must not ask for it.
    """
    model = _model(preset, charge_aware=False)
    assert constants.TOTAL_CHARGE not in model.compute_specs
    assert constants.TOTAL_CHARGE not in model.train_specs


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_film_model_declares_the_charge_input(preset):
    model = _model(preset)
    assert constants.TOTAL_CHARGE in model.compute_specs
    assert constants.ATOMS_TO_STRUCTURE_MAP in model.compute_specs


# -- the two identity guarantees ---------------------------------------------


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_film_is_the_identity_at_initialisation(preset):
    """gate = 0 at init, so a freshly built FiLM model ignores the charge.

    This is what lets a FiLM preset be loaded with pretrained base weights and
    reproduce the base model exactly.
    """
    model = _model(preset)
    calc = TPCalculator(model=model)
    at = _atoms()
    energies = [_run(calc, at, q)[0] for q in (-2.0, 0.0, 2.0)]
    assert max(energies) - min(energies) == 0.0
    assert _run(calc, at, 2.0)[2] == 0.0  # dE/dq identically zero


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_neutral_structure_is_unaffected_by_the_gate(preset):
    """gamma(0) = beta(0) = 0 exactly, because the MLP carries no biases.

    So a neutral structure gives the unconditioned energy whatever the FiLM
    weights have learned.
    """
    model = _model(preset)
    calc = TPCalculator(model=model)
    at = _atoms()
    e_closed = _run(calc, at, 0.0)[0]
    _set_gate(model, 0.5)
    e_open = _run(calc, at, 0.0)[0]
    assert e_open == e_closed


# -- derivatives -------------------------------------------------------------


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_dedq_matches_finite_difference(preset):
    model = _model(preset)
    calc = TPCalculator(model=model)
    at = _atoms()
    _set_gate(model, 0.5)

    h = 1e-5
    for q in (-1.5, 0.0, 1.5):
        wf = _run(calc, at, q)[2]
        fd = (_run(calc, at, q + h)[0] - _run(calc, at, q - h)[0]) / (2 * h)
        assert wf == pytest.approx(fd, abs=1e-7), f"at q={q}"

    # and the energy genuinely depends on the charge now
    energies = [_run(calc, at, q)[0] for q in (-2.0, 0.0, 2.0)]
    assert max(energies) - min(energies) > 1e-3


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_forces_remain_exact_at_nonzero_charge(preset):
    """FiLM must not disturb the d/d(bond_vector) route to the forces."""
    model = _model(preset)
    calc = TPCalculator(model=model)
    at = _atoms()
    _set_gate(model, 0.5)

    q, d = 1.5, 1e-5
    forces = _run(calc, at, q)[1]
    positions = at.get_positions()
    for i in range(2):
        for k in range(3):
            p = positions.copy()
            p[i, k] += d
            e_plus = _run(calc, at, q, p)[0]
            p = positions.copy()
            p[i, k] -= d
            e_minus = _run(calc, at, q, p)[0]
            assert -(e_plus - e_minus) / (2 * d) == pytest.approx(
                forces[i, k], abs=1e-5
            )


def test_charge_couples_to_geometry():
    """d2E/drdq != 0 -- the point of conditioning, and what makes Born
    effective charges reachable."""
    model = _model("GRACE_2LAYER_FILM")
    calc = TPCalculator(model=model)
    at = _atoms()
    _set_gate(model, 0.5)
    f_lo = _run(calc, at, -1.5)[1]
    f_hi = _run(calc, at, +1.5)[1]
    assert np.abs(f_hi - f_lo).max() > 1e-6


# -- serialization -----------------------------------------------------------


@pytest.mark.parametrize("preset", ["GRACE_1LAYER_FILM", "GRACE_2LAYER_FILM"])
def test_instructions_round_trip(preset, tmp_path):
    """`capture_init_args` stores a dotted __cls__ path, so the new instruction
    needs no registry entry to deserialize."""
    instructions = _instructions(preset)
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), instructions)

    text = path.read_text()
    assert "FiLMChargeScalar" in text
    # a FiLM instruction must appear after the tensor it consumes:
    # load_instructions resolves references in file order
    for consumed, film in (("rho", "rho_film"), ("I_1_LN", "I_1_film")):
        if film in text:
            assert text.index(f"\n  {consumed}:") < text.index(f"\n  {film}:")

    reloaded = load_instructions(str(path))
    assert set(reloaded) == set(instructions)
    film_names = [k for k in reloaded if "film" in k]
    assert film_names, "no FiLM instruction survived the round trip"
    for name in film_names:
        assert reloaded[name].n_out == instructions[name].n_out
        assert reloaded[name].lmax == 0
