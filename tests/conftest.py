import os
import sys
from typing import Iterator

import matplotlib
import pandas as pd
import pytest

# Ensure non-interactive backend for matplotlib during tests
matplotlib.use("Agg")

# Ensure project root is importable (so `import src.assistant` works)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def sales_csv_path() -> str:
    """Return the path to the sample sales CSV used by the app."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sales_data.csv")


@pytest.fixture(scope="session")
def df_sample(sales_csv_path: str) -> pd.DataFrame:
    """Load and return the sales data as a pandas DataFrame with parsed dates.

    Returns:
        pd.DataFrame: DataFrame with a parsed `Date` column.
    """
    df = pd.read_csv(sales_csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@pytest.fixture()
def chdir_tmp(tmp_path: "pytest.TempPathFactory") -> Iterator[None]:
    """Temporarily change the working directory to a temp path for tests.

    This helps isolate filesystem writes (e.g., images) during tests.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)



