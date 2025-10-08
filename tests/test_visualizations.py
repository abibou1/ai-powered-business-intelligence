from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.assistant import generate_visualizations


def test_generate_visualizations_saves_four_figures(df_sample: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, chdir_tmp: None) -> None:
    saved_paths: List[str] = []

    def _fake_savefig(path: str, *args: object, **kwargs: object) -> None:
        # capture requested save paths, but write to tmp files to avoid project images dir
        saved_paths.append(path)

    monkeypatch.setattr(plt, "savefig", _fake_savefig, raising=True)

    # Ensure Date is datetime to avoid errors in groupby by period
    df = df_sample.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    generate_visualizations(df)
    assert len(saved_paths) == 4



