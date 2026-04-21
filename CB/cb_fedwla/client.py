"""Federated Client Module for CatBoost."""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from .task import get_partition, calculate_uncertainty

RANDOM_SEARCH_SEED = 42


class FlowerClient(NumPyClient):
    """Federated client for CatBoost local hyperparameter optimisation."""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        partition_id,
        local_num_examples,
        local_data_quality,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.partition_id = partition_id
        self.local_num_examples = int(local_num_examples)
        self.local_data_quality = float(local_data_quality)

        self.rng = np.random.default_rng(RANDOM_SEARCH_SEED + partition_id)

        # Random initialisation coherent with Table 5.
        self.init_iterations = int(self.rng.integers(20, 101))
        self.init_depth = int(self.rng.integers(4, 9))
        self.init_lr = round(float(self.rng.uniform(0.05, 0.30)), 4)

    def _sanitise_params(self, params: np.ndarray) -> tuple[int, int, float]:
        """Clamp hyperparameters to valid CatBoost ranges."""
        iterations = max(10, int(np.round(float(params[0]))))
        depth = max(1, int(np.round(float(params[1]))))
        learning_rate = max(0.001, float(params[2]))
        return iterations, depth, learning_rate

    def _build_model(
        self,
        iterations: int,
        depth: int,
        learning_rate: float,
    ) -> CatBoostClassifier:
        """Create a CatBoost model with the fixed settings used in the study."""
        return CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            task_type="CPU",
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=RANDOM_SEARCH_SEED + self.partition_id,
        )

    def _fit_and_score(self, cfg: tuple[int, int, float]) -> tuple[CatBoostClassifier, float]:
        """Train a model and compute weighted F1 on the local evaluation split."""
        model = self._build_model(*cfg)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred = np.asarray(y_pred).reshape(-1)
        score = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        return model, float(score)

    def get_parameters(self, config):
        """Return the client's random initial hyperparameters."""
        hyperparams = np.array(
            [
                self.init_iterations,
                self.init_depth,
                self.init_lr,
            ],
            dtype=np.float32,
        )
        return [hyperparams]

    def fit(self, parameters, config):
        """Evaluate Hg first, then run local random search around it, and return H_i*."""
        global_params = parameters[0]
        global_cfg = self._sanitise_params(global_params)

        # Baseline evaluation of the received global hyperparameters.
        best_model, best_local_f1 = self._fit_and_score(global_cfg)
        best_local_config = global_cfg

        # Local bounded random search around the current global configuration.
        global_iters, global_depth, global_lr = global_cfg
        for _ in range(3):
            iters = int(self.rng.integers(max(10, global_iters - 10), global_iters + 11))
            depth = int(self.rng.integers(max(2, global_depth - 1), global_depth + 3))

            lr_min = max(0.01, global_lr * 0.8)
            lr_max = min(1.0, global_lr * 1.2)
            lr_ = float(self.rng.uniform(lr_min, lr_max))

            candidate_cfg = (iters, depth, lr_)
            candidate_model, candidate_f1 = self._fit_and_score(candidate_cfg)
            if candidate_f1 > best_local_f1:
                best_model = candidate_model
                best_local_f1 = candidate_f1
                best_local_config = candidate_cfg

        # The round winner is the only configuration sent to the server.
        self.model = best_model

        y_pred_final = self.model.predict(self.X_test)
        y_pred_final = np.asarray(y_pred_final).reshape(-1)
        final_accuracy = accuracy_score(self.y_test, y_pred_final)
        final_f1 = f1_score(self.y_test, y_pred_final, average="weighted", zero_division=0)

        uncertainty = calculate_uncertainty(self.model, self.X_test)

        metrics = {
            "accuracy": final_accuracy,
            "f1": final_f1,
            "uncertainty": uncertainty,
            "data_quality": self.local_data_quality,
        }

        print(
            f"[Client {self.partition_id}] fit => "
            f"acc={final_accuracy:.4f} f1={final_f1:.4f} "
            f"N={self.local_num_examples} Q={self.local_data_quality:.4f} U={uncertainty:.4f}"
        )

        best_local_params = [np.array(best_local_config, dtype=np.float32)]

        return best_local_params, self.local_num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate the global parameters sent by the server."""
        global_params = parameters[0]
        test_cfg = self._sanitise_params(global_params)

        temp_model = self._build_model(*test_cfg)
        temp_model.fit(self.X_train, self.y_train)

        y_pred = temp_model.predict(self.X_test)
        y_pred = np.asarray(y_pred).reshape(-1)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

        return 0.0, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }



def client_fn(context: Context):
    """Federated client constructor."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = context.run_config.get("num-partitions", 10)
    dataset_path = context.run_config.get("dataset-path", "")
    imbalance_factor = context.run_config.get("imbalance-factor", 5)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        local_num_examples,
        local_data_quality,
    ) = get_partition(partition_id, num_partitions, dataset_path, imbalance_factor)

    return FlowerClient(
        X_train,
        y_train,
        X_test,
        y_test,
        partition_id,
        local_num_examples,
        local_data_quality,
    ).to_client()


app = ClientApp(client_fn=client_fn)
