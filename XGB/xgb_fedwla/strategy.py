"""FedWLA Custom Aggregation Strategy Module."""

import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays

class FedWLA(FedAvg):
    """
    Federated Weighted Learning Aggregation (FedWLA).
    Aggregates hyperparameters based on:
      - Model uncertainty (entropy).
      - Local dataset quality (class balance).
      - Number of local training examples.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        total_weight = 0.0
        aggregated_params = None

        for client, fit_res in results:
            metrics = fit_res.metrics
            uncertainty = metrics.get("uncertainty", 1.0)
            data_quality = metrics.get("data_quality", 1.0)
            num_examples = fit_res.num_examples

            weight = data_quality * (1.0 / (uncertainty + 1e-6)) * float(num_examples)

            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays = [arr.astype(np.float64) for arr in client_ndarrays]

            if aggregated_params is None:
                aggregated_params = [weight * arr for arr in client_ndarrays]
            else:
                aggregated_params = [
                    agg + weight * arr
                    for agg, arr in zip(aggregated_params, client_ndarrays)
                ]

            total_weight += weight

        if total_weight > 0:
            aggregated_params = [param / total_weight for param in aggregated_params]
        else:
            return None, {}

        aggregated_params[0][0] = max(1, int(np.round(aggregated_params[0][0])))
        aggregated_params[0][1] = max(0.001, float(aggregated_params[0][1]))
        aggregated_params[0][2] = max(10, int(np.round(aggregated_params[0][2])))
        aggregated_params[0][3] = float(np.clip(aggregated_params[0][3], 0.1, 1.0))

        new_global_parameters = ndarrays_to_parameters(aggregated_params)

        best_f1 = max(r[1].metrics.get("f1", 0.0) for r in results)
        metrics_aggregated = {"best_f1_fit": best_f1}

        return new_global_parameters, metrics_aggregated
