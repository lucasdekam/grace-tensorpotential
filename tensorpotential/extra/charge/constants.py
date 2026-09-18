"""Data-dict and dataframe keys for charge conditioning.

Local to this subpackage, following ``extra/gen_tensor/constants.py``. The one
exception is ``TOTAL_CHARGE`` itself, which lives in the core
``tensorpotential.constants`` because ``calculator/asecalculator.py`` has to
name it to decide whether to attach the data builder -- the same treatment
``ATOMIC_MAGMOM`` / ``ATOMIC_POS`` / ``CELL_VECTORS`` already get.
"""

from typing import Final


# Predictions
PREDICT_WORK_FUNCTION: Final[str] = "work_function"
# -d2E/drdq, in raw dF/dq units (eV / (A e)). The (A eps0) conversion to
# Born-effective-charge units is deliberately NOT done in the graph -- see the
# note in databuilder.py.
PREDICT_DF_DQ: Final[str] = "df_dq"
# d2E/dq2, the inverse frozen-nuclei capacitance up to 1/A, in V/e. Free: it
# is the second source of the same gradient call that produces df_dq.
PREDICT_D2E_DQ2: Final[str] = "d2e_dq2"

# Labels
DATA_REFERENCE_WORK_FUNCTION: Final[str] = "true_work_function"
DATA_WORK_FUNCTION_WEIGHTS: Final[str] = "work_function_weight"
DATA_REFERENCE_DF_DQ: Final[str] = "true_df_dq"
DATA_DF_DQ_WEIGHTS: Final[str] = "df_dq_weight"
DATA_REFERENCE_D2E_DQ2: Final[str] = "true_d2e_dq2"
DATA_D2E_DQ2_WEIGHTS: Final[str] = "d2e_dq2_weight"

# Keys read out of ase.Atoms.info / .arrays
INFO_TOTAL_CHARGE: Final[str] = "total_charge"
INFO_WORK_FUNCTION: Final[str] = "work_function"
INFO_D2E_DQ2: Final[str] = "d2Edq2"
ARRAYS_DF_DQ: Final[str] = "df_dq"
# the Born-effective-charge form of the same label, Z* = (A eps0) dF/dq
ARRAYS_BEC_Z: Final[str] = "bec_z"

# vacuum permittivity in e^2 / (eV Angstrom); the one place it is written down
EPSILON_0: Final[float] = 0.005526349406
