"""Federated Server Module for LightGBM."""

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

    initial_ndarray = np.array([
        rng.integers(4, 9),
        round(float(rng.uniform(0.01, 0.1)), 4),
        rng.integers(100, 501),
        rng.integers(16, 31)
    ], dtype=np.float32)
    
    initial_parameters = ndarrays_to_parameters([initial_ndarray])

    strategy = FedWLA(
        fraction_fit=0.8,      
        fraction_evaluate=0.8,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
