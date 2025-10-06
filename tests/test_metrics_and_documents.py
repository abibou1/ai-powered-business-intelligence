from typing import List

import pandas as pd
from langchain.schema import Document

from src.assistant import compute_basic_metrics, build_documents


def test_compute_basic_metrics(df_sample: pd.DataFrame) -> None:
    metrics = compute_basic_metrics(df_sample)
    assert "total_sales" in metrics
    assert "total_sales_by_region" in metrics
    # Cross-check with pandas operations
    assert metrics["total_sales"] == df_sample["Sales"].sum()
    pd.testing.assert_series_equal(
        metrics["total_sales_by_region"],
        df_sample.groupby("Region")["Sales"].sum(),
        check_names=True,
    )


def test_build_documents_row_count(df_sample: pd.DataFrame) -> None:
    docs: List[Document] = build_documents(df_sample)
    assert len(docs) == len(df_sample)  # only row-level docs are enabled
    # Validate content fields exist in first doc
    sample = docs[0].page_content
    assert "Date:" in sample and "Product:" in sample and "Sales:" in sample



