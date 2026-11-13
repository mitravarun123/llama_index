import builtins
from typing import Any, List, Optional

import pytest

from llama_index.core.evaluation.notebook_utils import get_retrieval_results_df
from llama_index.core.evaluation.retrieval.base import RetrievalEvalResult
from llama_index.core.evaluation.retrieval.metrics_base import RetrievalMetricResult


pd = pytest.importorskip("pandas")


def make_result(**scores: float) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        query="local query",
        expected_ids=["expected"],
        retrieved_ids=["retrieved"],
        retrieved_texts=["local text"],
        metric_dict={
            name: RetrievalMetricResult(score=score) for name, score in scores.items()
        },
    )


@pytest.mark.parametrize("metric_keys", [None, []])
def test_retrieval_results_default_metrics(
    metric_keys: Optional[List[str]],
) -> None:
    names = ["baseline", "candidate"]
    results = [
        [make_result(hit_rate=1.0, mrr=0.5), make_result(hit_rate=0.0, mrr=0.0)],
        [make_result(hit_rate=1.0, mrr=0.25)],
    ]
    actual = get_retrieval_results_df(names, results, metric_keys)
    expected = pd.DataFrame(
        {
            "retrievers": names,
            "hit_rate": [0.5, 1.0],
            "mrr": [0.25, 0.25],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert len(results[0]) == 2 and len(results[1]) == 1
    assert results[0][0].metric_vals_dict == {"hit_rate": 1.0, "mrr": 0.5}
    assert metric_keys is None or metric_keys == []


def test_retrieval_results_explicit_metric_order() -> None:
    actual = get_retrieval_results_df(
        ["baseline", "candidate"],
        [
            [
                make_result(precision=0.25, recall=0.5),
                make_result(precision=0.75, recall=1.0),
            ],
            [make_result(precision=1.0, recall=0.5)],
        ],
        metric_keys=["recall", "precision"],
    )
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "retrievers": ["baseline", "candidate"],
                "recall": [0.75, 0.5],
                "precision": [0.5, 1.0],
            }
        ),
    )


def test_retrieval_results_ignore_unrequested_missing_metrics() -> None:
    actual = get_retrieval_results_df(
        ["baseline"],
        [[make_result(hit_rate=1.0, mrr=0.5), make_result(hit_rate=0.0)]],
        metric_keys=["hit_rate"],
    )
    assert actual.to_dict(orient="list") == {
        "retrievers": ["baseline"],
        "hit_rate": [0.5],
    }


@pytest.mark.parametrize(
    ("name_count", "group_count"), [(1, 2), (2, 1), (0, 1), (1, 0)]
)
def test_retrieval_results_mismatched_lengths(
    name_count: int, group_count: int
) -> None:
    with pytest.raises(ValueError, match="names and results_arr must have same length"):
        get_retrieval_results_df(
            [f"retriever-{index}" for index in range(name_count)],
            [[make_result(hit_rate=1.0, mrr=1.0)] for _ in range(group_count)],
        )


def test_retrieval_results_empty_named_group() -> None:
    with pytest.raises(ValueError, match="Retriever 'empty' has no evaluation results"):
        get_retrieval_results_df(
            ["baseline", "empty"], [[make_result(hit_rate=1.0, mrr=1.0)], []]
        )


@pytest.mark.parametrize("metric_keys", [None, [], ["mrr"]])
def test_retrieval_results_missing_metric_in_every_result(
    metric_keys: Optional[List[str]],
) -> None:
    with pytest.raises(
        ValueError,
        match="Retriever 'candidate', result index 0: missing requested metric 'mrr'",
    ):
        get_retrieval_results_df(
            ["candidate"],
            [[make_result(hit_rate=1.0), make_result(hit_rate=0.0)]],
            metric_keys,
        )


@pytest.mark.parametrize("missing_index", [0, 1])
@pytest.mark.parametrize("metric_keys", [None, [], ["mrr"]])
def test_retrieval_results_missing_metric_in_one_result(
    missing_index: int, metric_keys: Optional[List[str]]
) -> None:
    results = [make_result(hit_rate=1.0, mrr=0.5), make_result(hit_rate=0.0, mrr=0.0)]
    results[missing_index] = make_result(hit_rate=1.0)
    with pytest.raises(
        ValueError,
        match=f"Retriever 'candidate', result index {missing_index}: missing requested metric 'mrr'",
    ):
        get_retrieval_results_df(
            ["baseline", "candidate"],
            [[make_result(hit_rate=1.0, mrr=1.0)], results],
            metric_keys,
        )


@pytest.mark.parametrize("metric_keys", [None, [], ["precision"]])
def test_retrieval_results_empty_comparison(
    metric_keys: Optional[List[str]],
) -> None:
    actual = get_retrieval_results_df([], [], metric_keys)
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"retrievers": []}))


def test_retrieval_results_pandas_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_pandas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pandas":
            raise ImportError("pandas unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pandas)
    with pytest.raises(ImportError, match="pandas is required for this function"):
        get_retrieval_results_df([], [])
