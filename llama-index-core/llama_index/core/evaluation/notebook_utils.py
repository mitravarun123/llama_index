"""Notebook utils."""

from collections import defaultdict
from typing import Any, List, Optional, Tuple

from llama_index.core.evaluation import EvaluationResult
from llama_index.core.evaluation.retrieval.base import RetrievalEvalResult

DEFAULT_METRIC_KEYS = ["hit_rate", "mrr"]


def get_retrieval_results_df(
    names: List[str],
    results_arr: List[List[RetrievalEvalResult]],
    metric_keys: Optional[List[str]] = None,
) -> Any:
    """
    Display mean retrieval metrics for each named retriever.

    Each result must contain every requested metric. Both `None` and an empty
    `metric_keys` list select the default metrics. Retrievers may have different
    numbers of results; an empty comparison returns only the `retrievers` column.

    Raises:
        ValueError: If names and result groups have different lengths, a named
            group is empty, or a result is missing a requested metric. Missing
            metric errors identify the retriever and zero-based result index.

    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for this function. Please install it with `pip install pandas`."
        )

    if len(names) != len(results_arr):
        raise ValueError("names and results_arr must have same length.")

    metric_keys = metric_keys or DEFAULT_METRIC_KEYS

    avg_metrics_dict = defaultdict(list)
    for name, eval_results in zip(names, results_arr):
        if not eval_results:
            raise ValueError(f"Retriever {name!r} has no evaluation results.")

        metric_dicts = []
        for result_index, eval_result in enumerate(eval_results):
            metric_dict = eval_result.metric_vals_dict
            for metric_key in metric_keys:
                if metric_key not in metric_dict:
                    raise ValueError(
                        f"Retriever {name!r}, result index {result_index}: "
                        f"missing requested metric {metric_key!r}."
                    )
            metric_dicts.append(metric_dict)
        results_df = pd.DataFrame(metric_dicts)

        for metric_key in metric_keys:
            avg_metrics_dict[metric_key].append(results_df[metric_key].mean())

    return pd.DataFrame({"retrievers": names, **avg_metrics_dict})


def get_eval_results_df(
    names: List[str], results_arr: List[EvaluationResult], metric: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Organizes EvaluationResults into a deep dataframe and computes the mean
    score.

    result:
        result_df: pd.DataFrame representing all the evaluation results
        mean_df: pd.DataFrame of average scores groupby names
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for this function. Please install it with `pip install pandas`."
        )

    if len(names) != len(results_arr):
        raise ValueError("names and results_arr must have same length.")

    qs = []
    ss = []
    fs = []
    rs = []
    cs = []
    for res in results_arr:
        qs.append(res.query)
        ss.append(res.score)
        fs.append(res.feedback)
        rs.append(res.response)
        cs.append(res.contexts)

    deep_df = pd.DataFrame(
        {
            "rag": names,
            "query": qs,
            "answer": rs,
            "contexts": cs,
            "scores": ss,
            "feedbacks": fs,
        }
    )
    mean_df = pd.DataFrame(deep_df.groupby(["rag"])["scores"].mean()).T
    if metric:
        mean_df.index = [f"mean_{metric}_score"]

    return deep_df, mean_df
