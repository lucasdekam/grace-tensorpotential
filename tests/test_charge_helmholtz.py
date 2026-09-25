"""Tests for `HelmholtzChargeTarget` / `GRACE_2LAYER_HELMHOLTZ`.

The head imposes RAZOR's capacitor expansion,

    E(q) = E_0 + phi_0 q + q^2 / (2 C_0),   phi_0 = phi_ref + P_z / (eps0 A)

so unlike FiLM the *shape* of E(q) is a claim the tests can check exactly rather
than to a finite-difference tolerance. Tests 1 and 2 are the ones that pin the
construction: the first says E(q) really is quadratic with the stated curvature,
the second says P_z is a genuine z-polarization rather than an invariant that
memorised the slab orientation.

Note ASE's `check_state` compares positions/numbers/cell/pbc but NOT
`atoms.info`, so changing only the charge does not invalidate the calculator's
cache. Every evaluation goes through `_run`, which calls `calc.reset()`.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pytest
import tensorflow as tf
from ase.build import bulk

from tensorpotential.calculator import TPCalculator
from tensorpotential.extra.charge import constants as cc
from tensorpotential.extra.charge.model import (
    ComputeStructureEnergyForcesVirialCharge,
)
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.tpmodel import TPModel

PRESET = "GRACE_2LAYER_HELMHOLTZ"
ELEMENT_MAP = {"Al": 0}
# deliberately tiny -- these tests are about the charge algebra, not accuracy
SMALL = dict(rcut=5.0, n_mlp_dens=4, embedding_size=16, max_order=2, n_rad_base=6)

AREA = 82.31  # A^2, razor's cell, so the numbers are in a realistic range
PHI_REF = 5.75
INV_C = 9.16


def _instructions(**kwargs):
    kwargs.setdefault("lmax", [2, 2])
    kwargs.setdefault("n_rad_max", [8, 8])
    kwargs.setdefault("prod_func_n_max", [8, 8])
    kwargs.setdefault("indicator_lmax", 1)
    kwargs.setdefault("charge_area", AREA)
    kwargs.setdefault("charge_phi_ref", PHI_REF)
    kwargs.setdefault("charge_inv_capacitance", INV_C)
    return get_preset(PRESET)(
        element_map=ELEMENT_MAP, **{**SMALL, **kwargs}
    ).get_instructions()


def _model(predict_bec=False, seed=20260925, open_dipole=True, **kwargs):
    tf.random.set_seed(seed)
    model = TPModel(
        _instructions(**kwargs),
        compute_function=ComputeStructureEnergyForcesVirialCharge(
            predict_bec=predict_bec
        ),
    )
    model.build(tf.float64)
    # `dipole_scale` is zero-initialised, so P_z vanishes on a fresh model.
    # Everything except the initialisation tests needs it nonzero; nothing needs
    # it trained, only nonzero and geometry-dependent.
    if open_dipole:
        _set_dipole_scale(model, 1.0)
    return model


def _atoms():
    at = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 1, 1)
    at.rattle(0.05, seed=1)
    return at


def _run(calc, at, q):
    """Energy, forces and dE/dq at charge `q`."""
    a = at.copy()
    a.info[cc.INFO_TOTAL_CHARGE] = float(q)
    a.calc = calc
    calc.reset()  # atoms.info is invisible to ASE's check_state
    energy = a.get_potential_energy()
    forces = a.get_forces()
    wf = calc.outputs[0][cc.PREDICT_WORK_FUNCTION].numpy().ravel()[0]
    return energy, forces, wf


def _set_dipole_scale(model, value):
    """The polarization branch is gated to zero at init; open it."""
    n = 0
    for v in model.variables:
        if "dipole_scale" in v.name:
            v.assign(tf.fill(v.shape, tf.constant(value, dtype=v.dtype)))
            n += 1
    assert n == 1, f"expected one dipole_scale variable, found {n}"
    return n


# -- 1. the expansion is exactly what it claims to be ------------------------


def test_energy_is_exactly_quadratic_in_the_charge():
    """E(q) - E(0) - q phi_0 == q^2 / (2 C_0), to float64 round-off.

    Not a finite-difference check: the head asserts an exact functional form, so
    anything beyond round-off means the form is not what the docstring says.
    """
    calc = TPCalculator(model=_model())
    at = _atoms()
    e0, _, phi0 = _run(calc, at, 0.0)
    for q in (-1.5, -0.5, 0.25, 1.0):
        e, _, _ = _run(calc, at, q)
        assert e - e0 - q * phi0 == pytest.approx(0.5 * INV_C * q**2, abs=1e-9)


def test_work_function_is_exactly_affine_in_the_charge():
    """dE/dq(q) - dE/dq(0) == q / C_0, with the slope the constructor was given."""
    calc = TPCalculator(model=_model())
    at = _atoms()
    _, _, phi0 = _run(calc, at, 0.0)
    for q in (-1.5, -0.5, 0.25, 1.0):
        _, _, phi = _run(calc, at, q)
        assert phi - phi0 == pytest.approx(INV_C * q, abs=1e-9)


def test_d2edq2_is_the_capacitance_parameter():
    """The autograd second derivative returns 1/C_0 itself, at any q."""
    model = _model(predict_bec=True)
    calc = TPCalculator(model=model)
    at = _atoms()
    for q in (-1.0, 0.0, 1.0):
        a = at.copy()
        a.info[cc.INFO_TOTAL_CHARGE] = float(q)
        a.calc = calc
        calc.reset()
        a.get_potential_energy()
        d2 = calc.outputs[0][cc.PREDICT_D2E_DQ2].numpy().ravel()[0]
        assert d2 == pytest.approx(INV_C, abs=1e-9)


def test_forces_are_exactly_linear_in_the_charge():
    """F(q) = F_0 + q dF/dq with dF/dq independent of q.

    The capacitive term carries no geometry, so the charge enters the forces
    only through P_z -- which is what makes the SEBEC q-independent.
    """
    calc = TPCalculator(model=_model())
    at = _atoms()
    _, f0, _ = _run(calc, at, 0.0)
    _, f1, _ = _run(calc, at, 1.0)
    dfdq = f1 - f0
    for q in (-1.5, 0.5, 2.0):
        _, f, _ = _run(calc, at, q)
        np.testing.assert_allclose(f, f0 + q * dfdq, atol=1e-9)


# -- 2. equivariance: P_z is a polarization, not a memorised constant --------


def test_rotation_about_the_slab_normal_changes_nothing():
    calc = TPCalculator(model=_model())
    at = _atoms()
    rot = at.copy()
    rot.rotate(37.0, "z", rotate_cell=True)

    for q in (-1.0, 0.0, 1.0):
        e_a, _, phi_a = _run(calc, at, q)
        e_b, _, phi_b = _run(calc, rot, q)
        assert e_b == pytest.approx(e_a, abs=1e-8)
        assert phi_b == pytest.approx(phi_a, abs=1e-8)


def test_mirroring_the_slab_flips_the_polarization():
    """Under z -> -z, E_0 and 1/C_0 are unchanged but P_z must flip sign.

    This is the test a scalar (invariant) readout cannot pass: every invariant
    descriptor is mirror-even, so an invariant head would return the *same*
    phi_0 and could only ever memorise one slab orientation. `Parity.SCALAR` is
    `[[0, 1]]`, so E_0 really is mirror-even and the whole difference sits in
    the l=1 branch.
    """
    calc = TPCalculator(model=_model())
    at = _atoms()
    mirrored = at.copy()
    pos = mirrored.get_positions()
    pos[:, 2] *= -1.0
    mirrored.set_positions(pos)
    mirrored.wrap()

    for q in (-1.0, 0.5):
        e_a, _, phi_a = _run(calc, at, q)
        e_b, _, phi_b = _run(calc, mirrored, q)
        # P_z / (eps0 A) is what is left after removing the two constants
        pol_a = phi_a - PHI_REF - q * INV_C
        pol_b = phi_b - PHI_REF - q * INV_C
        assert abs(pol_a) > 1e-6, "polarization is zero; the test proves nothing"
        assert pol_b == pytest.approx(-pol_a, abs=1e-8)
        # the charge-free energy is mirror-invariant
        assert e_b - q * pol_b == pytest.approx(e_a - q * pol_a, abs=1e-8)


# -- 3. padding ---------------------------------------------------------------


@pytest.mark.parametrize("pad", [2, 4, 11])
def test_padding_does_not_move_the_energy_or_the_work_function(pad):
    """Exact padding invariance, which FiLM does not have.

    `p_z` vanishes on a padded atom already; the charge-only bracket is masked
    and divided by the real-atom count, so exactly N_real atoms each contribute
    1/N_real of it whatever the padding width.

    `pad_atoms_number` must be a positive int or None, so the reference is the
    narrowest legal padding rather than none at all.
    """
    model = _model()
    ref = TPCalculator(model=model, pad_atoms_number=1)
    tst = TPCalculator(model=model, pad_atoms_number=pad)
    at = _atoms()
    for q in (-1.0, 0.0, 1.0):
        e_r, f_r, phi_r = _run(ref, at, q)
        e_t, f_t, phi_t = _run(tst, at, q)
        assert e_t == pytest.approx(e_r, abs=1e-9)
        assert phi_t == pytest.approx(phi_r, abs=1e-9)
        np.testing.assert_allclose(f_t, f_r, atol=1e-9)


# -- 4. initialisation --------------------------------------------------------


def test_a_fresh_model_is_exactly_the_bare_capacitor():
    """`dipole_scale` is zero-initialised, so P_z vanishes until it opens.

    Without this the randomly-initialised reduce emits a large arbitrary P_z --
    5.7 V RMS on the work function against a ~1.2 V label spread -- and the
    carefully chosen `phi_ref` would be pointless.
    """
    calc = TPCalculator(model=_model(open_dipole=False))
    at = _atoms()
    for q in (-1.0, 0.0, 0.75):
        _, _, phi = _run(calc, at, q)
        assert phi == pytest.approx(PHI_REF + INV_C * q, abs=1e-9)


def test_opening_the_dipole_scale_makes_the_polarization_nonzero():
    """Guards the test above from being vacuous in the other direction."""
    at = _atoms()
    closed = TPCalculator(model=_model(open_dipole=False))
    opened = TPCalculator(model=_model(open_dipole=True))
    _, _, phi_c = _run(closed, at, 0.0)
    _, _, phi_o = _run(opened, at, 0.0)
    assert abs(phi_o - phi_c) > 1e-3


def test_phi_ref_and_inv_capacitance_are_trainable_by_default():
    model = _model()
    names = {v.name: v for v in model.trainable_variables}
    assert any("phi_ref" in n for n in names), "phi_ref is not trainable"
    assert any("inv_capacitance" in n for n in names), "1/C_0 is not trainable"


def test_the_two_scalars_can_be_frozen():
    model = _model(charge_train_phi_ref=False, charge_train_inv_capacitance=False)
    names = [v.name for v in model.trainable_variables]
    assert not any("phi_ref" in n for n in names)
    assert not any("inv_capacitance" in n for n in names)


# -- 5. the derivatives agree with finite differences ------------------------


def test_dedq_matches_finite_difference():
    calc = TPCalculator(model=_model())
    at = _atoms()
    h, q = 1e-4, 0.3
    e_p, _, _ = _run(calc, at, q + h)
    e_m, _, _ = _run(calc, at, q - h)
    _, _, phi = _run(calc, at, q)
    assert phi == pytest.approx((e_p - e_m) / (2 * h), abs=1e-7)


def test_forces_are_exact_at_nonzero_charge():
    calc = TPCalculator(model=_model())
    at = _atoms()
    q, h = 0.7, 1e-5
    _, f, _ = _run(calc, at, q)
    pos = at.get_positions()
    for i in (0, 3):
        for c in range(3):
            up, dn = pos.copy(), pos.copy()
            up[i, c] += h
            dn[i, c] -= h
            a_up, a_dn = at.copy(), at.copy()
            a_up.set_positions(up)
            a_dn.set_positions(dn)
            e_p, _, _ = _run(calc, a_up, q)
            e_m, _, _ = _run(calc, a_dn, q)
            assert f[i, c] == pytest.approx(-(e_p - e_m) / (2 * h), abs=1e-6)


def test_dfdq_matches_central_difference_of_the_forces():
    model = _model(predict_bec=True)
    calc = TPCalculator(model=model)
    at = _atoms()
    h, q = 1e-4, 0.2
    _, f_p, _ = _run(calc, at, q + h)
    _, f_m, _ = _run(calc, at, q - h)
    a = at.copy()
    a.info[cc.INFO_TOTAL_CHARGE] = q
    a.calc = calc
    calc.reset()
    a.get_potential_energy()
    dfdq = calc.outputs[0][cc.PREDICT_DF_DQ].numpy()[: len(at)]
    np.testing.assert_allclose(dfdq, (f_p - f_m) / (2 * h), atol=1e-6)


# -- 6. serialization ---------------------------------------------------------


def test_instructions_round_trip(tmp_path):
    instructions = _instructions()
    path = tmp_path / "instructions.yaml"
    save_instructions_dict(str(path), instructions)
    text = path.read_text()
    assert "HelmholtzChargeTarget" in text
    assert "FunctionReduceParticular" in text

    # the head must appear after the reduces it consumes: load_instructions
    # resolves references in file order
    for consumed in ("I_dipole_lin", "I_dipole_gated"):
        assert text.index(f"\n  {consumed}:") < text.index("\n  HelmholtzChargeTarget:")

    loaded = load_instructions(str(path))
    assert set(loaded) == set(instructions)
    assert "I_dipole_lin" in loaded

    head = loaded["HelmholtzChargeTarget"]
    assert head.area == AREA
    assert head.slab_normal_axis == 2
    assert head.phi_ref_init == PHI_REF
    assert head.inv_capacitance_init == INV_C


def test_area_is_required():
    with pytest.raises(AssertionError, match="charge_area"):
        get_preset(PRESET)(element_map=ELEMENT_MAP, **SMALL)
