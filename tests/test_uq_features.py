import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest
import tensorflow as tf
from tensorpotential.uq.feature_extraction import (
    FeatureBuffer,
    extract_features,
    extract_features_bulk,
    batch_feature_chunks,
    setup_feature_calculator,
)
from tensorpotential.uq import constants as uqc


def test_feature_buffer():
    """Test FeatureBuffer capacity growth and slicing."""
    dim = 4
    buf = FeatureBuffer(feature_dim=dim, capacity=10)

    # Fill partially
    f1 = np.ones((6, dim))
    e1 = np.zeros(6, dtype=np.int32)
    buf.append(f1, e1)
    assert len(buf) == 6

    # Trigger growth
    f2 = np.ones((10, dim)) * 2
    e2 = np.ones(10, dtype=np.int32)
    buf.append(f2, e2)
    assert len(buf) == 16
    assert buf._capacity >= 16

    # Verify contents
    assert np.allclose(buf.features[6:16], f2)
    assert np.all(buf.elements[6:16] == 1)

    # Test iteration
    chunks = list(buf.iter_chunks(chunk_size=7))
    assert len(chunks) == 3  # 7, 7, 2
    assert chunks[0][0].shape == (7, dim)
    assert chunks[2][0].shape == (2, dim)


def test_extract_features(uq_setup):
    """Test feature extraction generators with a mocked-checkpoint model."""
    calc = setup_feature_calculator(
        uq_setup["model_yaml"],
        uq_setup["checkpoint"],
        feature_spec=uq_setup["feature_spec"],
    )
    atoms = uq_setup["atoms"][:3]

    # 1. extract_features generator
    gen = extract_features(calc, atoms, element_map=uq_setup["element_map"])
    results = list(gen)
    assert len(results) == 3
    assert results[0][0].shape[1] == uq_setup["feature_dim"]

    # 2. extract_features_bulk collector
    all_f, all_e = extract_features_bulk(
        calc, atoms, element_map=uq_setup["element_map"]
    )
    total_at = sum(len(at) for at in atoms)
    assert all_f.shape == (total_at, uq_setup["feature_dim"])
    assert all_e.shape == (total_at,)


def test_batch_feature_chunks():
    """Test re-batching of per-structure features into fixed-size chunks."""

    def mock_gen():
        yield np.ones((3, 2)), np.zeros(3)
        yield np.ones((5, 2)), np.zeros(5)
        yield np.ones((2, 2)), np.zeros(2)

    chunks = list(batch_feature_chunks(mock_gen(), chunk_size=7))
    # 1st chunk gathers 3 + 5 = 8 atoms
    # 2nd chunk gathers remaining 2 atoms
    assert len(chunks) == 2
    assert chunks[0][0].shape == (8, 2)
    assert chunks[1][0].shape == (2, 2)


def test_feature_transform_contract(uq_setup):
    """Pin the feature-transform contract.

    - ``make_basis_rp_spec`` defaults to ``DEFAULT_FEATURE_TRANSFORM`` (``None`` —
      the production uqv6 feature uses no element-wise transform), so the
      ``uq_feature_transform`` key is omitted by default.
    - ``feature_transform="asinh"`` (the uqv4 feature) stamps the key verbatim.
    - ``feature_transform=None`` (default / legacy linear) omits the key entirely
      so old artifacts keep their exact key set.
    - ``_basis_rp_spec_from_artifact`` reads the key back; an absent key -> None.
    - The instruction applies element-wise asinh (vs identity for None) and
      rejects unknown transforms.
    """
    from tensorpotential.uq.factories import (
        make_basis_rp_spec,
        _basis_rp_spec_from_artifact,
    )
    from tensorpotential.uq.constants import DEFAULT_FEATURE_TRANSFORM
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    # The default transform is None (uqv6) -> key omitted.
    assert DEFAULT_FEATURE_TRANSFORM is None
    spec = make_basis_rp_spec(uq_setup["model_yaml"], rp_dim=16)
    assert uqc.UQ_FEATURE_TRANSFORM not in spec
    # Explicit asinh (uqv4) stamps the key ...
    spec_asinh = make_basis_rp_spec(
        uq_setup["model_yaml"], rp_dim=16, feature_transform="asinh"
    )
    assert str(np.asarray(spec_asinh[uqc.UQ_FEATURE_TRANSFORM]).item()) == "asinh"
    # ... and explicit linear omits the key (legacy artifacts).
    spec_linear = make_basis_rp_spec(
        uq_setup["model_yaml"], rp_dim=16, feature_transform=None
    )
    assert uqc.UQ_FEATURE_TRANSFORM not in spec_linear

    # Round-trip through the artifact reader (uses .extra_data on the model).
    class _Stub:
        def __init__(self, extra):
            self.extra_data = extra

    asinh_spec = _basis_rp_spec_from_artifact(_Stub(spec_asinh))
    assert asinh_spec["transform"] == "asinh"
    default_spec = _basis_rp_spec_from_artifact(_Stub(spec))
    assert default_spec["transform"] is None  # absent key -> None
    linear_spec = _basis_rp_spec_from_artifact(_Stub(spec_linear))
    assert linear_spec["transform"] is None  # absent key -> legacy linear

    # The instruction applies asinh element-wise vs identity for None.
    x = tf.constant([[-3.0, 0.0, 2.0, 50.0]], dtype=tf.float64)
    asinh_instr = RandomProjectedBasisFeatures([], out_dim=4, feature_transform="asinh")
    asinh_instr.projection = tf.eye(4, dtype=tf.float64)  # stub dtype carrier
    np.testing.assert_allclose(
        asinh_instr._apply_feature_transform(x).numpy(), np.arcsinh(x.numpy())
    )
    linear_instr = RandomProjectedBasisFeatures([], out_dim=4, feature_transform=None)
    np.testing.assert_array_equal(
        linear_instr._apply_feature_transform(x).numpy(), x.numpy()
    )

    # Unknown transforms are rejected at construction.
    import pytest

    with pytest.raises(ValueError, match="feature_transform must be one of"):
        RandomProjectedBasisFeatures([], out_dim=4, feature_transform="rmsblock_asinh")


def test_uqv6_zero_basis_gradient_is_finite_under_xla(uq_setup):
    """The uqv6 norm channels must not emit NaN gradients for a zero basis.

    A padded batch always contains one: the fake atom's bonds are all dummies
    beyond the cutoff, so the envelope zeros its whole basis. Asserts the
    gradient is finite everywhere and that padded bonds contribute exactly
    nothing — see ``RandomProjectedBasisFeatures._safe_norm`` for why the norm
    must not be ``tf.norm``.
    """
    from tensorpotential.uq.factories import patch_instructions_for_basis_rp_features
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.tpmodel import execute_instructions
    from tensorpotential import constants as tc
    from .utils import build_tf_batch

    instructions = load_instructions(uq_setup["model_yaml"])
    patch_instructions_for_basis_rp_features(
        instructions, out_dim=16, normalize=True, add_density_channel=True
    )
    tp = TensorPotential(instructions, param_dtype=tf.float64)
    tp.load_checkpoint(uq_setup["checkpoint"])

    data, _, n_bonds_real = build_tf_batch(
        uq_setup["atoms"][0],
        uq_setup["element_map"],
        tp.model.compute_specs,
        pad_atoms=4,
        pad_bonds=8,
    )

    @tf.function(jit_compile=True)
    def masked_feature_grad(data):
        data = dict(data)  # execute_instructions writes into the dict
        with tf.GradientTape() as tape:
            tape.watch(data[tc.BOND_VECTOR])
            execute_instructions(data, instructions, False)
            features = data[uqc.FEATURES]
            real = tf.cast(
                tf.range(tf.shape(features)[0]) < data[tc.N_ATOMS_BATCH_REAL],
                features.dtype,
            )
            # Same shape as the HAL objective: mask padded atoms multiplicatively.
            return tape.gradient(
                tf.reduce_sum(features * real[:, None]), data[tc.BOND_VECTOR]
            )

    grad = masked_feature_grad(data).numpy()
    bad = np.flatnonzero(~np.isfinite(grad).all(axis=1))
    assert bad.size == 0, (
        f"non-finite feature gradient on bond rows {bad[:10]} "
        f"of {len(grad)} ({n_bonds_real} real, rest padded)"
    )
    # Dummy bonds sit beyond the cutoff, so they must contribute exactly nothing.
    np.testing.assert_array_equal(grad[n_bonds_real:], 0.0)


def test_batched_feature_iterator(uq_setup):
    """Test the optimized batched iterator (requires StreamingDatasetWrapper logic)."""
    from tensorpotential.uq.feature_extraction import batched_feature_iterator
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tensorpot import TensorPotential

    # We need a decorated model for this iterator
    from tensorpotential.uq.factories import patch_instructions_for_basis_rp_features

    instructions = load_instructions(uq_setup["model_yaml"])
    # UQ feature = basis-RP projection of the invariant energy-path basis,
    # written under the canonical FEATURES key by the appended instruction.
    patch_instructions_for_basis_rp_features(
        instructions, out_dim=uq_setup["feature_spec"]["out_dim"]
    )

    tp = TensorPotential(instructions, param_dtype=tf.float64)
    tp.load_checkpoint(uq_setup["checkpoint"])
    tp.model.decorate_compute_function()
    # Ensure the compute function returns the feature in the output dict
    tp.model.compute_function.extra_return_keys = [uqc.FEATURES]

    atoms = uq_setup["atoms"][:5]
    it = batched_feature_iterator(
        atoms, tp.model, element_map=uq_setup["element_map"], cutoff=6.0, verbose=False
    )

    results = list(it)
    assert len(results) > 0
    total_collected = sum(len(f) for f, e, w in results)
    total_expected = sum(len(at) for at in atoms)
    assert total_collected == total_expected
    assert results[0][0].shape[1] == uq_setup["feature_dim"]
    # Default weight is 1.0 when atoms.info has no UQ_WEIGHT tag.
    assert all(np.all(w == 1.0) for _, _, w in results)


# ---------------------------------------------------------------------------
# _NORM_FLOOR: the constant guarding the zero-basis gradient.
#
# The model-level test above runs fp64, but every shipped uqv6 SavedModel is
# fp32, and the floor's whole justification is an fp32 property (a subnormal
# constant is flushed to zero under XLA). These exercise the norm directly at
# both precisions so a future "tighten the floor" edit cannot regress fp32
# while fp64-only tests stay green.
# ---------------------------------------------------------------------------

_ZERO_AND_NORMAL = [[0.0, 0.0, 0.0], [0.3, -0.4, 0.5]]  # padded/isolated atom, then a real one


def _safe_norm_grad(dtype, floor):
    """d/dx sum(_safe_norm(x)) under XLA, for a batch whose first row is all zeros."""
    from types import SimpleNamespace

    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    me = SimpleNamespace(_NORM_FLOOR=floor)

    @tf.function(jit_compile=True)
    def grad(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            total = tf.reduce_sum(RandomProjectedBasisFeatures._safe_norm(me, x))
        return tape.gradient(total, x)

    return grad(tf.constant(_ZERO_AND_NORMAL, dtype=dtype)).numpy()


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["fp32", "fp64"])
def test_safe_norm_gradient_is_finite_at_a_zero_row(dtype):
    """The production floor must work at fp32, the precision the models ship in."""
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    grad = _safe_norm_grad(dtype, RandomProjectedBasisFeatures._NORM_FLOOR)
    assert np.isfinite(grad).all(), f"non-finite gradient at {dtype.name}: {grad}"


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["fp32", "fp64"])
def test_safe_norm_floor_is_load_bearing(dtype):
    """Without the floor the zero row differentiates to 0/0 = NaN under XLA.

    Guards against someone "simplifying" the floor away: if this ever passes with
    floor 0, the jit no longer needs the guard and the comment should be revisited.
    """
    grad = _safe_norm_grad(dtype, 0.0)
    assert not np.isfinite(grad[0]).all(), (
        f"expected NaN at {dtype.name} with no floor, got {grad[0]} -- "
        "if XLA now guards this itself, revisit _NORM_FLOOR"
    )


def test_production_floor_clears_the_fp32_subnormal_boundary():
    """1e-38 is below float32's smallest NORMAL value (1.18e-38), so XLA flushes it
    to zero and the NaN returns. Measured boundary: fails at 1e-38, holds at 1e-37.
    The shipped 1e-30 keeps several orders of margin -- do not tighten it."""
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    assert not np.isfinite(_safe_norm_grad(tf.float32, 1e-38)[0]).all(), (
        "1e-38 unexpectedly survived; the subnormal-flush boundary moved"
    )
    assert np.isfinite(_safe_norm_grad(tf.float32, 1e-37)[0]).all()
    assert RandomProjectedBasisFeatures._NORM_FLOOR >= 1e-37, (
        "floor dropped to or below the fp32 subnormal boundary"
    )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["fp32", "fp64"])
def test_safe_norm_leaves_an_ordinary_row_alone(dtype):
    """The floor must be invisible for any atom with a non-zero basis."""
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    v = np.array(_ZERO_AND_NORMAL[1])
    expected = v / np.linalg.norm(v)  # d|x|/dx = x/|x|
    grad = _safe_norm_grad(dtype, RandomProjectedBasisFeatures._NORM_FLOOR)
    tol = 1e-6 if dtype == tf.float32 else 1e-12
    np.testing.assert_allclose(grad[1], expected, atol=tol)
