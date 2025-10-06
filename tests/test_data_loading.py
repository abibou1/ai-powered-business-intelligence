from typing import Any

import pandas as pd
import pytest

import os

from src.assistant import load_data, ensure_openai_env


def test_load_data_success(df_sample: pd.DataFrame) -> None:
    df = load_data("data/sales_data.csv")
    assert not df.empty
    assert {"Date", "Product", "Region", "Sales"}.issubset(df.columns)


def test_ensure_openai_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY_TEST", "dummy")

    def _fake_getenv(key: str, default: str | None = None) -> str | None:
        if key == "OPENAI_API_KEY":
            return os.environ.get("OPENAI_API_KEY_TEST", default)
        return os.environ.get(key, default)

    monkeypatch.setattr(os, "getenv", _fake_getenv, raising=True)
    assert ensure_openai_env() == "dummy"


def test_ensure_openai_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY_TEST", raising=False)

    def _fake_getenv_missing(key: str, default: str | None = None) -> str | None:
        if key == "OPENAI_API_KEY":
            return None
        return os.environ.get(key, default)

    monkeypatch.setattr(os, "getenv", _fake_getenv_missing, raising=True)
    with pytest.raises(ValueError):
        ensure_openai_env()



