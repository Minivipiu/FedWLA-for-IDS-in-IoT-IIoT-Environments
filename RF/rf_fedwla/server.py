"""Federated Server Module for Random Forest."""

import numpy as np
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters

from .strategy import FedWLA
from .metrics import weighted_average

HP_INIT_SEED = 42

def server_fn(context: Context):
    """Initialise and configure the ServerApp."""
    
    num_rounds = context.run_config.get("num-server-rounds", 10)

    rng = np.random.default_rng(HP_INIT_SEED)

    # Initial global parameters aligned with the paper
    initial_ndarray = np.array([
        rng.integers(50, 151),                  # n_estimators
        rng.integers(10, 51),                   # max_depth
        rng.integers(2, 11),                    # min_samples_split
        rng.integers(1, 6)                      # min_samples_leaf
    ], dtype=np.float32)
    
    initial_parameters = ndarrays_to_parameters([initial_ndarray])

    strategy = FedWLA(
        fraction_fit=1,      
        fraction_evaluate=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
