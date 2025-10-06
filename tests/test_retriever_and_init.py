from typing import Any, Dict, List

import pandas as pd
import pytest

import src.assistant as assistant


class _DummyRetriever:
    def __init__(self, docs: List[Any]) -> None:
        self._docs = docs

    def invoke(self, query: str) -> List[Any]:  # pragma: no cover - trivial
        return self._docs


class _DummyVectorStore:
    def __init__(self, docs: List[Any]) -> None:
        self._docs = docs

    def add_documents(self, docs: List[Any]) -> None:  # pragma: no cover - trivial
        self._docs.extend(docs)

    def as_retriever(self, **_: Any) -> _DummyRetriever:
        return _DummyRetriever(self._docs)


class _DummyDoc:
    def __init__(self, content: str, meta: Dict[str, Any] | None = None) -> None:
        self.page_content = content
        self.metadata = meta or {}


def test_build_retriever_from_documents_monkeypatched(df_sample: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    documents = assistant.build_documents(df_sample.head(3))

    class _FakeEmbeddings:  # pragma: no cover - trivial
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def _fake_chroma(**kwargs: Any) -> _DummyVectorStore:
        # simulate persistent vector store without disk/network
        return _DummyVectorStore(docs=[])

    monkeypatch.setattr(assistant, "OpenAIEmbeddings", _FakeEmbeddings, raising=True)
    monkeypatch.setattr(assistant, "Chroma", _fake_chroma, raising=True)

    retriever = assistant.build_retriever_from_documents(documents)
    out = retriever.invoke("query")
    assert isinstance(out, list)


def test_initialize_components_with_mocks(monkeypatch: pytest.MonkeyPatch, df_sample: pd.DataFrame) -> None:
    # avoid env and network
    monkeypatch.setattr(assistant, "ensure_openai_env", lambda: "ok", raising=True)
    monkeypatch.setattr(assistant, "load_data", lambda: df_sample, raising=True)

    class _FakeLLM:  # pragma: no cover - trivial
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(assistant, "get_llm", lambda temperature=0: _FakeLLM(), raising=True)

    # Provide Document-like objects so initialize_components' test_retriever doesn't break
    dummy_retriever = _DummyRetriever([_DummyDoc("sample content", {"row_index": 0})])
    monkeypatch.setattr(assistant, "build_retriever_from_documents", lambda docs: dummy_retriever, raising=True)

    sentinel_qa = object()
    sentinel_agent = object()
    sentinel_conv = object()

    monkeypatch.setattr(assistant, "build_qa_chain", lambda llm, r: sentinel_qa, raising=True)
    monkeypatch.setattr(assistant, "build_pandas_agent", lambda llm, df: sentinel_agent, raising=True)
    monkeypatch.setattr(assistant, "build_conversational_chain", lambda llm, r: sentinel_conv, raising=True)

    qa_chain, agent, memory, conv_chain, df = assistant.initialize_components()
    assert qa_chain is sentinel_qa
    assert agent is sentinel_agent
    assert conv_chain is sentinel_conv
    assert isinstance(df, pd.DataFrame)
    assert memory is not None



