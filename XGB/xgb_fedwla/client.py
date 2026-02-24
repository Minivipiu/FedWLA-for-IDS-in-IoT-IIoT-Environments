"""Federated Client Module."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from .task import get_partition, calculate_uncertainty, calculate_balance_quality

RANDOM_SEARCH_SEED = 42

class FlowerClient(NumPyClient):
    """Federated client for local hyperparameter optimisation using Random Search."""
    
    def __init__(self, X_train, y_train, X_test, y_test, partition_id) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.partition_id = partition_id
        
        self.num_classes = len(np.unique(self.y_train))
        
        self.rng = np.random.default_rng(RANDOM_SEARCH_SEED + partition_id)
        
        self.best_max_depth = int(self.rng.integers(4, 9))
        self.best_lr = round(float(self.rng.uniform(0.01, 0.1)), 4)
        self.best_n_estimators = int(self.rng.integers(100, 501))
        self.best_subsample = round(float(self.rng.uniform(0.7, 1.0)), 2)
        
        self.best_f1 = 0.0
        self.model = None

    def get_parameters(self, config):
        """Return the current best local hyperparameters."""
        hyperparams = np.array([
            self.best_max_depth,
            self.best_lr,
            self.best_n_estimators,
            self.best_subsample
        ], dtype=np.float32)
        return [hyperparams]

    def fit(self, parameters, config):
        """Receive global parameters, perform local random search, and return the best."""
        global_params = parameters[0]
        global_max_depth = int(global_params[0])
        global_lr = float(global_params[1])
        global_n_estimators = int(global_params[2])
        global_subsample = float(global_params[3])

        best_local_f1 = -1.0
        best_local_config = (global_max_depth, global_lr, global_n_estimators, global_subsample)
        
        for _ in range(3):
            md = int(self.rng.integers(max(1, global_max_depth - 3), global_max_depth + 4))
            lr_ = float(self.rng.uniform(global_lr * 0.5, global_lr * 1.5))
            ne = int(self.rng.integers(max(10, global_n_estimators - 20), global_n_estimators + 21))
            
            ss_min = max(0.1, global_subsample * 0.8)
            ss_max = min(1.0, global_subsample * 1.2)
            ss = float(self.rng.uniform(ss_min, ss_max))

            temp_model = xgb.XGBClassifier(
                objective="multi:softmax", num_class=self.num_classes, booster="gbtree",
                max_depth=md, learning_rate=lr_, n_estimators=ne, subsample=ss,
                colsample_bytree=0.8, eval_metric="mlogloss",
                n_jobs=-1, random_state=RANDOM_SEARCH_SEED + self.partition_id
            )
            temp_model.fit(self.X_train, self.y_train)
            
            y_pred_local = temp_model.predict(self.X_test)
            local_f1 = f1_score(self.y_test, y_pred_local, average="weighted", zero_division=0)

            if local_f1 > best_local_f1:
                best_local_f1 = local_f1
                best_local_config = (md, lr_, ne, ss)

        if best_local_f1 > self.best_f1:
            self.best_f1 = best_local_f1
            (self.best_max_depth, self.best_lr, self.best_n_estimators, self.best_subsample) = best_local_config

        self.model = xgb.XGBClassifier(
            objective="multi:softmax", num_class=self.num_classes, booster="gbtree",
            max_depth=self.best_max_depth, learning_rate=self.best_lr,
            n_estimators=self.best_n_estimators, subsample=self.best_subsample,
            colsample_bytree=0.8, eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_SEARCH_SEED + self.partition_id
        )
        self.model.fit(self.X_train, self.y_train)

        y_pred_final = self.model.predict(self.X_test)
        
        final_accuracy = accuracy_score(self.y_test, y_pred_final)
        final_f1 = f1_score(self.y_test, y_pred_final, average="weighted", zero_division=0)
        
        uncertainty = calculate_uncertainty(self.model, self.X_test)
        data_quality = calculate_balance_quality(self.y_train)

        metrics = {
            "accuracy": final_accuracy,
            "f1": final_f1,
            "uncertainty": uncertainty,
            "data_quality": data_quality
        }

        print(f"[Client {self.partition_id}] fit => acc={final_accuracy:.4f} f1={final_f1:.4f}")

        best_local_params = [np.array([
            self.best_max_depth, self.best_lr, self.best_n_estimators, self.best_subsample
        ], dtype=np.float32)]
        
        return best_local_params, len(self.X_train), metrics

    def evaluate(self, parameters, config):
        """Evaluate the global parameters sent by the server."""
        global_params = parameters[0]
        test_max_depth = max(1, int(global_params[0]))
        test_lr = max(0.001, float(global_params[1]))
        test_n_estimators = max(10, int(global_params[2]))
        test_subsample = max(0.1, min(1.0, float(global_params[3])))

        temp_model = xgb.XGBClassifier(
            objective="multi:softmax", num_class=self.num_classes, booster="gbtree",
            max_depth=test_max_depth, learning_rate=test_lr, n_estimators=test_n_estimators,
            subsample=test_subsample, colsample_bytree=0.8,
            eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_SEARCH_SEED + self.partition_id
        )
        
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
            "f1": f1
        }

def client_fn(context: Context):
    """Client App constructor."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = context.run_config.get("num-partitions", 10)
    dataset_path = context.run_config.get("dataset-path", "")
    imbalance_factor = context.run_config.get("imbalance-factor", 5)

    X_train, y_train, X_test, y_test = get_partition(
        partition_id, num_partitions, dataset_path, imbalance_factor
    )
    
    return FlowerClient(X_train, y_train, X_test, y_test, partition_id).to_client()

app = ClientApp(client_fn=client_fn)
