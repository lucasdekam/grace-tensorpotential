import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorpotential.calculator import TPCalculator, predict_structures

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()


def set_magmom(at, magmoms):
    magmoms = np.array(magmoms)
    assert len(at) == magmoms.shape[0]
    if magmoms.shape == (len(at), 3):
        at.arrays["initial_magmoms"] = magmoms
    elif (magmoms.shape == (len(at), 1)) or (magmoms.shape == (len(at),)):
        new_magmoms = np.zeros((len(at), 3))
        new_magmoms[:, 2] = magmoms
        at.arrays["initial_magmoms"] = new_magmoms
    else:
        raise ValueError("mag_mom shape is not recognized")
    return at


def _atoms_with_magmoms(df):
    """The structures to evaluate, with initial magnetic moments applied.

    `predict_structures` copies each structure before attaching a calculator, so
    the dataframe's own objects can be handed over as-is. Only the magmom branch
    needs a copy of its own, because `set_magmom` writes into `atoms.arrays` —
    copying the whole column up front would pin a second copy of the dataset for
    the entire run.
    """
    if "mag_mom" not in df.columns:
        return df["ase_atoms"].tolist()
    return [
        at if mm is None else set_magmom(at.copy(), mm)
        for at, mm in zip(df["ase_atoms"], df["mag_mom"])
    ]


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        help="provide path to the saved_model directory",
        type=str,
        default="saved_model",
        dest="model_path",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="path to the dataset.pkl.gzip containing ase_atoms structures",
        type=str,
        default="dataset.pkl.gz",
        dest="dataset_file",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="path to the OUTPUT dataset (pkl.gzip) containing energy_predicted and forces_predicted",
        type=str,
        default="predicted_dataset.pkl.gz",
        dest="output",
    )

    parser.add_argument(
        "-e",
        "--raise-errors",
        help="Whether to NOT ignore errors and stop the program.",
        action="store_true",
        default=False,
        dest="raise_errors",
    )

    args_parse = parser.parse_args(args)

    model_path = os.path.abspath(args_parse.model_path)
    dataset_file = args_parse.dataset_file
    output_file = args_parse.output
    raise_errors = args_parse.raise_errors

    logger.info(f"Loading model from: {model_path}")
    calc = TPCalculator(
        model=model_path,
        pad_atoms_number=20,
        pad_neighbors_fraction=0.30,
        # max_number_reduction_recompilation=3,
    )

    logger.info(f"Loading dataset from: {dataset_file}")
    df = pd.read_pickle(dataset_file, compression="gzip")

    logger.info("Starting prediction")

    # predict_structures evaluates largest-first (one XLA compile for the widest
    # shape) and returns results in the dataframe's own order.
    with tqdm(total=len(df)) as bar:
        pred = predict_structures(
            _atoms_with_magmoms(df),
            calc,
            properties=("energy", "forces", "stress"),
            on_error="raise" if raise_errors else "warn",
            progress=lambda done, total: bar.update(),
        )
    df["energy_predicted"] = pred["energy"]
    df["forces_predicted"] = pred["forces"]
    df["stress_predicted"] = pred["stress"]

    logger.info(f"Saving dataset to {output_file}")
    df.drop(columns=["ase_atoms"]).to_pickle(output_file, compression="gzip")


if __name__ == "__main__":
    main()
