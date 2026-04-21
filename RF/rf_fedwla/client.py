"""Federated Client Module for Random Forest."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from .task import get_partition, calculate_uncertainty

RANDOM_SEARCH_SEED = 42


class FlowerClient(NumPyClient):
    """Federated client for Random Forest local hyperparameter optimisation."""

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

        # Random initialisation, coherent with the manuscript.
        self.init_n_estimators = int(self.rng.integers(50, 151))
        self.init_max_depth = int(self.rng.integers(10, 51))
        self.init_min_samples_split = int(self.rng.integers(2, 11))
        self.init_min_samples_leaf = int(self.rng.integers(1, 6))

    def _sanitise_params(self, params: np.ndarray) -> tuple[int, int, int, int]:
        """Clamp hyperparameters to valid Random Forest ranges."""
        n_estimators = max(10, int(np.round(float(params[0]))))
        max_depth = max(1, int(np.round(float(params[1]))))
        min_samples_split = max(2, int(np.round(float(params[2]))))
        min_samples_leaf = max(1, int(np.round(float(params[3]))))
        return n_estimators, max_depth, min_samples_split, min_samples_leaf

    def _build_model(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
    ) -> RandomForestClassifier:
        """Create a Random Forest model with the fixed settings used in the study."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion="entropy",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEARCH_SEED + self.partition_id,
        )

    def _fit_and_score(self, cfg: tuple[int, int, int, int]) -> tuple[RandomForestClassifier, float]:
        """Train a model and compute weighted F1 on the local evaluation split."""
        model = self._build_model(*cfg)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        score = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        return model, float(score)

    def get_parameters(self, config):
        """Return the client's random initial hyperparameters."""
        hyperparams = np.array(
            [
                self.init_n_estimators,
                self.init_max_depth,
                self.init_min_samples_split,
                self.init_min_samples_leaf,
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
        global_ne, global_md, global_mss, global_msl = global_cfg
        for _ in range(3):
            ne = int(self.rng.integers(max(10, global_ne - 20), global_ne + 21))
            md = int(self.rng.integers(max(1, global_md - 3), global_md + 4))
            mss = int(self.rng.integers(max(2, global_mss - 2), global_mss + 3))
            msl = int(self.rng.integers(max(1, global_msl - 2), global_msl + 3))

            candidate_cfg = (ne, md, mss, msl)
            candidate_model, candidate_f1 = self._fit_and_score(candidate_cfg)
            if candidate_f1 > best_local_f1:
                best_model = candidate_model
                best_local_f1 = candidate_f1
                best_local_config = candidate_cfg

        self.model = best_model

        y_pred_final = self.model.predict(self.X_test)
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

        # Important: num_examples equals |D_local|, coherent with FedWLA Eq. 7.
        return best_local_params, self.local_num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate the global parameters sent by the server."""
        global_params = parameters[0]
        test_cfg = self._sanitise_params(global_params)

        temp_model = self._build_model(*test_cfg)
        temp_model.fit(self.X_train, self.y_train)

        y_pred = temp_model.predict(self.X_test)

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
    """Client App constructor."""
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
