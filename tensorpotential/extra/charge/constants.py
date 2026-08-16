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
# -(A eps0) d2E/drdq, in raw dF/dq units (eV / (A e)). The (A eps0) conversion
# to Born-effective-charge units is deliberately NOT done in the graph -- see
# the note in databuilder.py.
PREDICT_DF_DQ: Final[str] = "df_dq"

# Labels
DATA_REFERENCE_WORK_FUNCTION: Final[str] = "true_work_function"
DATA_WORK_FUNCTION_WEIGHTS: Final[str] = "work_function_weight"
DATA_REFERENCE_DF_DQ: Final[str] = "true_df_dq"
DATA_DF_DQ_WEIGHTS: Final[str] = "df_dq_weight"

# Keys read out of ase.Atoms.info / .arrays
INFO_TOTAL_CHARGE: Final[str] = "total_charge"
INFO_WORK_FUNCTION: Final[str] = "work_function"
ARRAYS_DF_DQ: Final[str] = "df_dq"
