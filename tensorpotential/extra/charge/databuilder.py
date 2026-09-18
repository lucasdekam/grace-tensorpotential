"""Supply the per-structure total charge (a model *input*) and the optional
work-function label.

No change to `cli/data.py` is needed to use plain extxyz: `load_dataset` already
dispatches `.xyz`/`.extxyz` to `load_extxyz`, which keeps the whole ASE object
in the `ase_atoms` column, so `atoms.info` survives verbatim and
`extract_from_row` can read it.

Enable in input.yaml with::

    data:
      extra_components:
        TotalChargeDataBuilder: {}

BORN EFFECTIVE CHARGES
----------------------
The model predicts raw `dF/dq` in eV/(A e). Datasets usually carry the
dimensionless Born form instead, `Z* = (A eps0) dF/dq`, and the `(A eps0)`
conversion happens **here, in Python, next to the data** rather than inside the
TF graph. The surface area is a per-dataset convention that needs an explicit
slab-normal axis, and computing it in two places is exactly how a silent 2.54x
error arose in a sibling project: for a slab whose normal is `a`, the surface
is spanned by `b` and `c`, not by `a` and `b`. One conversion site --
`bec_normal_axis` names the axis, and nothing downstream of this file knows
about areas.
"""

import logging

import numpy as np

from tensorpotential import constants
from tensorpotential.data.databuilder import AbstractDataBuilder, get_padding_dims
from tensorpotential.extra.charge import constants as cc

log = logging.getLogger()


class _RunningStats:
    """min/max/mean/std from running aggregates.

    Deliberately not a list of values: `extract_from_ase_atoms` is also called
    once per structure by the ASE calculator during MD, where nothing ever
    drains it, so anything that grows per structure is a leak.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.lo = float("inf")
        self.hi = float("-inf")

    def add(self, x):
        x = float(x)
        self.n += 1
        self.total += x
        self.total_sq += x * x
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)

    def summary(self, unit):
        mean = self.total / self.n
        var = max(self.total_sq / self.n - mean * mean, 0.0)
        return (
            f"{self.lo:+.3f} .. {self.hi:+.3f} {unit} "
            f"(mean {mean:+.3f}, std {var**0.5:.3f})"
        )


class TotalChargeDataBuilder(AbstractDataBuilder):
    """Per-structure `total_charge` input, plus an optional `work_function` label.

    Parameters
    ----------
    fit_work_function : bool
        Also emit the `work_function` label and its weight. Off by default, so
        the builder can be used purely to feed the charge into FiLM.
    default_charge : float
        Used when a structure carries no `total_charge` in `atoms.info`. The
        default of 0.0 means an existing uncharged dataset works unchanged.
    """

    def __init__(
        self,
        fit_work_function: bool = False,
        fit_df_dq: bool = False,
        fit_d2e_dq2: bool = False,
        default_charge: float = 0.0,
        charge_key: str = cc.INFO_TOTAL_CHARGE,
        work_function_key: str = cc.INFO_WORK_FUNCTION,
        df_dq_key: str = cc.ARRAYS_BEC_Z,
        d2e_dq2_key: str = cc.INFO_D2E_DQ2,
        bec_normal_axis: int | None = 2,
        normalize_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fit_work_function = fit_work_function
        self.fit_df_dq = fit_df_dq
        self.fit_d2e_dq2 = fit_d2e_dq2
        self.default_charge = default_charge
        self.charge_key = charge_key
        self.work_function_key = work_function_key
        self.df_dq_key = df_dq_key
        self.d2e_dq2_key = d2e_dq2_key
        if bec_normal_axis not in (0, 1, 2, None):
            raise ValueError(
                f"bec_normal_axis must be 0, 1, 2 or None, got {bec_normal_axis!r}"
            )
        self.bec_normal_axis = bec_normal_axis
        self.normalize_weights = normalize_weights
        # reported once per dataset in postprocess_dataset, then reset -- the
        # same builder instance handles train and test
        self._q_stats = _RunningStats()
        self._wf_stats = _RunningStats()
        self._bec_stats = _RunningStats()
        self._d2_stats = _RunningStats()
        self._n_defaulted = 0

    def _bec_to_raw(self, z_star, ase_atoms):
        """Z* -> dF/dq, the quantity the graph actually predicts.

        Z* = (A eps0) dF/dq with A the area of the surface the slab normal is
        perpendicular to. `bec_normal_axis=None` says the label is already raw
        dF/dq and skips the conversion entirely.
        """
        if self.bec_normal_axis is None:
            return z_star
        cell = np.asarray(ase_atoms.get_cell())
        u, v = [k for k in (0, 1, 2) if k != self.bec_normal_axis]
        area = np.linalg.norm(np.cross(cell[u], cell[v]))
        if area <= 0:
            raise ValueError(
                f"{self.df_dq_key} needs a surface area, but the cell spanned by "
                f"axes {u} and {v} has none. Set bec_normal_axis to the slab "
                f"normal, or to None if the label is already raw dF/dq."
            )
        return z_star / (area * cc.EPSILON_0)

    def _from_atoms(self, ase_atoms):
        if self.charge_key in ase_atoms.info:
            charge = float(ase_atoms.info[self.charge_key])
        else:
            charge = self.default_charge
            self._n_defaulted += 1
        self._q_stats.add(charge)

        res = {constants.TOTAL_CHARGE: np.array(charge).reshape(-1, 1)}
        if self.fit_work_function:
            if self.work_function_key not in ase_atoms.info:
                raise KeyError(
                    f"fit_work_function=True but atoms.info has no "
                    f"'{self.work_function_key}'. Charge conditioning alone does "
                    f"not need it -- set fit_work_function=False."
                )
            wf = float(ase_atoms.info[self.work_function_key])
            self._wf_stats.add(wf)
            res[cc.DATA_REFERENCE_WORK_FUNCTION] = np.array(wf).reshape(-1, 1)
            res[cc.DATA_WORK_FUNCTION_WEIGHTS] = np.ones((1, 1))

        if self.fit_df_dq:
            if self.df_dq_key not in ase_atoms.arrays:
                raise KeyError(
                    f"fit_df_dq=True but atoms.arrays has no '{self.df_dq_key}'. "
                    f"Charge conditioning alone does not need it -- set "
                    f"fit_df_dq=False."
                )
            z = np.asarray(ase_atoms.arrays[self.df_dq_key], dtype=float)
            if z.shape != (len(ase_atoms), 3):
                raise ValueError(
                    f"'{self.df_dq_key}' has shape {z.shape}, expected "
                    f"{(len(ase_atoms), 3)}"
                )
            for x in z.ravel():
                self._bec_stats.add(x)
            res[cc.DATA_REFERENCE_DF_DQ] = self._bec_to_raw(z, ase_atoms)
            res[cc.DATA_DF_DQ_WEIGHTS] = np.ones((len(ase_atoms), 1))

        if self.fit_d2e_dq2:
            if self.d2e_dq2_key not in ase_atoms.info:
                raise KeyError(
                    f"fit_d2e_dq2=True but atoms.info has no "
                    f"'{self.d2e_dq2_key}'."
                )
            d2 = float(ase_atoms.info[self.d2e_dq2_key])
            self._d2_stats.add(d2)
            res[cc.DATA_REFERENCE_D2E_DQ2] = np.array(d2).reshape(-1, 1)
            res[cc.DATA_D2E_DQ2_WEIGHTS] = np.ones((1, 1))
        return res

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        return self._from_atoms(ase_atoms)

    def extract_from_row(self, row, **kwarg):
        return self._from_atoms(row[constants.COLUMN_ASE_ATOMS])

    def get_sample_dtypes(self):
        return {k: self.float_dtype for k in self._keys()}

    def get_batch_dtypes(self):
        return self.get_sample_dtypes()

    def _struct_keys(self):
        """Keys with one row per structure."""
        keys = [constants.TOTAL_CHARGE]
        if self.fit_work_function:
            keys += [cc.DATA_REFERENCE_WORK_FUNCTION, cc.DATA_WORK_FUNCTION_WEIGHTS]
        if self.fit_d2e_dq2:
            keys += [cc.DATA_REFERENCE_D2E_DQ2, cc.DATA_D2E_DQ2_WEIGHTS]
        return keys

    def _atom_keys(self):
        """Keys with one row per atom -- padded and normalised by atom count,
        not by structure count, exactly like the force label."""
        return (
            [cc.DATA_REFERENCE_DF_DQ, cc.DATA_DF_DQ_WEIGHTS]
            if self.fit_df_dq
            else []
        )

    def _keys(self):
        return self._struct_keys() + self._atom_keys()

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}
        for key in self._struct_keys():
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = (
                np.concatenate(data_list, axis=0)
                .reshape(-1, 1)
                .astype(self.float_dtype)
            )
        # the df_dq label is [n_atoms, 3]; reshaping it to a column would
        # silently interleave the components
        for key in self._atom_keys():
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = np.concatenate(data_list, axis=0).astype(self.float_dtype)
        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        pad_nat, _, pad_nstruct = get_padding_dims(batch, max_pad_dict)
        # Pad with 0, i.e. the dummy structures are neutral and the dummy atoms
        # carry zero weight. For the charge this is not cosmetic: pad_batch in
        # GeometricalDataBuilder maps padded ATOMS to structure index
        # max_structs - 1, so FiLM's gather(q, map_atoms_to_structure) reads
        # these slots and they must be present and finite.
        if pad_nstruct > 0:
            for key in self._struct_keys():
                batch[key] = np.pad(
                    batch[key],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        if pad_nat > 0:
            for key in self._atom_keys():
                batch[key] = np.pad(
                    batch[key],
                    ((0, pad_nat), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

    def report(self):
        """Log what was actually read, then reset for the next dataset.

        The counts line is the one that earns its place: a missing
        `total_charge` silently becomes `default_charge`, and since it is a
        model *input* rather than a label, training on a quietly uncharged
        dataset looks perfectly healthy from the loss curves.

        The ranges are the extrapolation envelope -- FiLM sees the raw charge,
        so anything outside them is out of distribution. And the work-function
        std is the RMSE a mean predictor would score, which is what makes
        `rmse/wf` interpretable the moment it appears.
        """
        n = self._q_stats.n
        if n == 0:
            return

        if self._n_defaulted:
            log.warning(
                f"charge conditioning: {n - self._n_defaulted}/{n} structures carry "
                f"'{self.charge_key}' in atoms.info; {self._n_defaulted} defaulted to "
                f"{self.default_charge}"
            )
        else:
            log.info(
                f"charge conditioning: {self.charge_key} read from atoms.info "
                f"for {n}/{n} structures"
            )

        line = f"  {self.charge_key} {self._q_stats.summary('e')}"
        if self.fit_work_function and self._wf_stats.n:
            line += f" | {self.work_function_key} {self._wf_stats.summary('V')}"
        log.info(line)
        # reported in the units they were *read* in, before the (A eps0)
        # conversion, so the line can be checked against the source file
        if self.fit_df_dq and self._bec_stats.n:
            log.info(f"  {self.df_dq_key} {self._bec_stats.summary('e')}"
                     f"  (normal axis {self.bec_normal_axis})")
        if self.fit_d2e_dq2 and self._d2_stats.n:
            log.info(f"  {self.d2e_dq2_key} {self._d2_stats.summary('V/e')}")

        self._q_stats.reset()
        self._wf_stats.reset()
        self._bec_stats.reset()
        self._d2_stats.reset()
        self._n_defaulted = 0

    def postprocess_dataset(self, batches):
        self.report()
        if not self.normalize_weights:
            return
        active = []
        if self.fit_work_function:
            active.append(cc.DATA_WORK_FUNCTION_WEIGHTS)
        if self.fit_df_dq:
            active.append(cc.DATA_DF_DQ_WEIGHTS)
        if self.fit_d2e_dq2:
            active.append(cc.DATA_D2E_DQ2_WEIGHTS)
        for key in active:
            weight_sum = np.sum([np.sum(b[key]) for b in batches])
            if weight_sum > 0:
                for b in batches:
                    b[key] /= weight_sum
