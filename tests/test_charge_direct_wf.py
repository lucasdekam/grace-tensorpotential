"""Tests for `DirectWorkFunction` / `GRACE_2LAYER_FILM_WF`.

The head reads a work function out of the FiLM-conditioned invariant features
through its own MLP and a mean over real atoms,

    Phi(R, q) = phi_ref + s q + scale * <w_i(R, q)>,

so, unlike every other charge head here, Phi is NOT dE/dq. The tests pin that
split: the compute function must report the head as `work_function` and the
autograd derivative separately as `de_dq`, and a plain FiLM model must export
exactly what it did before.

ASE's `check_state` ignores `atoms.info`, so every evaluation goes through
`_run`, which resets the calculator.
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
from tensorpotential.extra.charge.model import ComputeStructureEnergyForcesVirialCharge
from tensorpotential.instructions import load_instructions, save_instructions_dict
from tensorpotential.potentials import get_preset
from tensorpotential.tpmodel import TPModel

ELEMENT_MAP = {"Al": 0}
SMALL = dict(rcut=5.0, n_mlp_dens=4, embedding_size=16, max_order=2, n_rad_base=6,
             lmax=[2, 2], n_rad_max=[8, 8], prod_func_n_max=[8, 8], indicator_lmax=1)
PHI_REF = 4.9
SLOPE = 9.16


def _instructions(preset="GRACE_2LAYER_FILM_WF", **kwargs):
    if preset == "GRACE_2LAYER_FILM_WF":
        kwargs.setdefault("wf_phi_ref", PHI_REF)
        kwargs.setdefault("wf_charge_slope", SLOPE)
    return get_preset(preset)(element_map=ELEMENT_MAP, **{**SMALL, **kwargs}).get_instructions()


def _open(model, name, value):
    n = 0
    for v in model.variables:
        if name in v.name:
            v.assign(tf.fill(v.shape, tf.constant(value, dtype=v.dtype)))
            n += 1
    assert n >= 1, f"no variable matching {name!r}"


def _model(preset="GRACE_2LAYER_FILM_WF", open_heads=True, seed=20260929, **kwargs):
    tf.random.set_seed(seed)
    model = TPModel(_instructions(preset, **kwargs),
                    compute_function=ComputeStructureEnergyForcesVirialCharge())
    model.build(tf.float64)
    if open_heads:
        _open(model, "gate", 0.5)          # FiLM: let the charge reach the features
        if preset == "GRACE_2LAYER_FILM_WF":
            _open(model, "wf_scale", 1.0)  # the direct head's zero-init scale
    return model


def _atoms():
    at = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 1, 1)
    at.rattle(0.05, seed=1)
    return at


def _run(calc, at, q):
    a = at.copy()
    a.info[cc.INFO_TOTAL_CHARGE] = float(q)
    a.calc = calc
    calc.reset()
    e = a.get_potential_energy()
    out = calc.outputs[0]
    wf = float(out[cc.PREDICT_WORK_FUNCTION].numpy().ravel()[0])
    dedq = float(out[cc.PREDICT_DE_DQ].numpy().ravel()[0]) if cc.PREDICT_DE_DQ in out else None
    return e, a.get_forces(), wf, dedq


def test_a_fresh_model_is_exactly_the_prior():
    """wf_scale starts at zero, so Phi = phi_ref + s q whatever the geometry."""
    calc = TPCalculator(model=_model(open_heads=False))
    for q in (-1.0, 0.0, 0.5):
        _, _, wf, _ = _run(calc, _atoms(), q)
        assert wf == pytest.approx(PHI_REF + SLOPE * q, abs=1e-10)


def test_the_head_depends_on_geometry_and_charge():
    calc = TPCalculator(model=_model())
    at = _atoms()
    at2 = at.copy()
    at2.rattle(0.1, seed=7)
    _, _, wf_a, _ = _run(calc, at, 0.3)
    _, _, wf_b, _ = _run(calc, at2, 0.3)
    _, _, wf_c, _ = _run(calc, at, -0.3)
    assert abs(wf_a - wf_b) > 1e-6
    # beyond the prior's linear term: the pooled readout itself sees q through FiLM
    assert abs((wf_a - wf_c) - SLOPE * 0.6) > 1e-6


def test_work_function_is_the_head_and_de_dq_is_the_energy_derivative():
    """The two are reported separately, and de_dq is the energy's derivative."""
    calc = TPCalculator(model=_model())
    at, q, h = _atoms(), 0.2, 1e-4
    _, _, wf, dedq = _run(calc, at, q)
    e_p, _, _, _ = _run(calc, at, q + h)
    e_m, _, _, _ = _run(calc, at, q - h)
    assert dedq is not None
    assert dedq == pytest.approx((e_p - e_m) / (2 * h), rel=1e-5, abs=1e-6)
    assert abs(wf - dedq) > 1e-3


@pytest.mark.parametrize("pad", [2, 4, 11])
def test_padding_does_not_move_the_work_function(pad):
    """The mean runs over real atoms only, so padding cannot shift Phi."""
    model = _model()
    ref = TPCalculator(model=model, pad_atoms_number=1)
    tst = TPCalculator(model=model, pad_atoms_number=pad)
    for q in (-1.0, 0.0, 1.0):
        _, _, wf_r, _ = _run(ref, _atoms(), q)
        _, _, wf_t, _ = _run(tst, _atoms(), q)
        assert wf_t == pytest.approx(wf_r, abs=1e-9)


def test_rotation_leaves_the_work_function_unchanged():
    calc = TPCalculator(model=_model())
    at = _atoms()
    rot = at.copy()
    rot.rotate(37, "x", rotate_cell=True)
    rot.rotate(71, "z", rotate_cell=True)
    _, _, wf, _ = _run(calc, at, 0.4)
    _, _, wf_rot, _ = _run(calc, rot, 0.4)
    assert wf_rot == pytest.approx(wf, abs=1e-8)


def test_a_plain_film_model_is_unchanged():
    """No head: work_function is dE/dq and no de_dq output appears."""
    calc = TPCalculator(model=_model("GRACE_2LAYER_FILM"))
    at, q, h = _atoms(), 0.2, 1e-4
    _, _, wf, dedq = _run(calc, at, q)
    e_p, _, _, _ = _run(calc, at, q + h)
    e_m, _, _, _ = _run(calc, at, q - h)
    assert dedq is None
    assert wf == pytest.approx((e_p - e_m) / (2 * h), rel=1e-5, abs=1e-6)


def test_the_energy_is_the_film_models():
    """Adding the head changes nothing about the energy or forces (same seed)."""
    base = TPCalculator(model=_model("GRACE_2LAYER_FILM"))
    wf = TPCalculator(model=_model())
    for q in (-0.5, 0.5):
        e_b, f_b, _, _ = _run(base, _atoms(), q)
        e_w, f_w, _, _ = _run(wf, _atoms(), q)
        assert e_w == pytest.approx(e_b, abs=1e-10)
        np.testing.assert_allclose(f_w, f_b, atol=1e-10)


def test_instructions_round_trip(tmp_path):
    instructions = _instructions()
    path = tmp_path / "instructions.yaml"
    save_instructions_dict(str(path), instructions)
    text = path.read_text()
    assert "DirectWorkFunction" in text
    # the head must come after the readout it pools
    assert text.index("\n  wf_readout:") < text.index(f"\n  {cc.PREDICT_WORK_FUNCTION_DIRECT}:")
    loaded = load_instructions(str(path))
    head = loaded[cc.PREDICT_WORK_FUNCTION_DIRECT]
    assert head.phi_ref_init == PHI_REF
    assert head.charge_slope_init == SLOPE
