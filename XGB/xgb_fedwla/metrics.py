"""Metrics aggregation module for the federated server."""

from typing import List, Tuple, Dict

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Calculate the weighted average of the metrics sent by the clients."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    metric_keys = set().union(*(m.keys() for _, m in metrics))

    for key in metric_keys:
        try:
            aggregated[key] = sum(
                num_examples * m.get(key, 0.0) for num_examples, m in metrics
            ) / total_examples
        except ZeroDivisionError:
            aggregated[key] = 0.0

    return aggregated
