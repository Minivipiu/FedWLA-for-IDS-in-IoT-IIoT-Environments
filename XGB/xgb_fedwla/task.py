"""Tasks module for data partitioning, caching, and local metrics calculation."""

import pickle
import tempfile
from pathlib import Path
from filelock import FileLock
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

PARTITION_SEED = 42

def create_noniid_partitions_weighted(
    df: pd.DataFrame, num_clients: int, label_col: str = "Attack",
    imbalance_factor: int = 5
):
    """Partition the dataframe guaranteeing all classes are present in all clients."""
    rng = np.random.default_rng(PARTITION_SEED)
    y = df[label_col].values
    classes = np.unique(y)

    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = idx_by_class[c]
        n_c_original = len(idx_c)
        if n_c_original == 0:
            continue

        required_min = num_clients * 2
        if n_c_original < required_min:
            extra_needed = required_min - n_c_original
            extra_idx = rng.choice(idx_c, size=extra_needed, replace=True)
            idx_c = np.concatenate([idx_c, extra_idx])

        rng.shuffle(idx_c)
        n_c = len(idx_c)

        counts = np.full(num_clients, 2, dtype=int)
        remaining = n_c - counts.sum()

        if remaining > 0:
            raw_weights = rng.integers(low=1, high=imbalance_factor + 1, size=num_clients).astype(float)
            weights = raw_weights / raw_weights.sum()
            extra_counts = np.floor(weights * remaining).astype(int)
            counts += extra_counts

            diff = remaining - extra_counts.sum()
            for i in range(abs(diff)):
                idx_client = i % num_clients
                counts[idx_client] += 1 if diff > 0 else -1

        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            if end > start:  
                client_indices[k].extend(idx_c[start:end])
            start = end

    return [df.iloc[np.sort(idxs)].copy() for idxs in client_indices]

def get_partition(partition_id: int, num_partitions: int, dataset_path: str, imbalance_factor: int):
    """Load and cache dataset securely using FileLock for parallel simulation."""
    cache_dir = Path(tempfile.gettempdir()) / "fedwla_cache"
    cache_dir.mkdir(exist_ok=True)
    
    dataset_name = Path(dataset_path).stem
    cache_prefix = cache_dir / f"part_{dataset_name}_{num_partitions}_{imbalance_factor}_{PARTITION_SEED}"
    lock_file = cache_dir / f"data_prep_{dataset_name}_{num_partitions}_{imbalance_factor}_{PARTITION_SEED}.lock"
    target_cache_file = Path(f"{cache_prefix}_{partition_id}.pkl")

    with FileLock(lock_file):
        if not target_cache_file.exists():
            data = pd.read_csv(dataset_path, low_memory=False)
            data = data.dropna()

            partitions = create_noniid_partitions_weighted(
                df=data, num_clients=num_partitions, label_col="Attack",
                imbalance_factor=imbalance_factor
            )
            
            for i, p in enumerate(partitions):
                with open(f"{cache_prefix}_{i}.pkl", "wb") as f:
                    pickle.dump(p, f)

    with open(target_cache_file, "rb") as f:
        partition = pickle.load(f)
    
    train, test = train_test_split(
        partition, 
        test_size=0.2, 
        random_state=PARTITION_SEED,
        stratify=partition["Attack"]
    )
    
    X_train = train.drop(columns=["Attack"], errors='ignore').values
    y_train = train["Attack"].astype(int).values
    X_test = test.drop(columns=["Attack"], errors='ignore').values
    y_test = test["Attack"].astype(int).values
    
    return X_train, y_train, X_test, y_test

def calculate_uncertainty(model: xgb.XGBClassifier, X: np.ndarray) -> float:
    """Calculate the average entropy of the predictions."""
    try:
        probabilities = model.predict_proba(X)
        entropies = []
        for p in probabilities:
            p = np.clip(p, 1e-12, 1.0)
            ent = -np.sum(p * np.log2(p))
            entropies.append(ent)
        return float(np.mean(entropies))
    except Exception:
        return 1.0

def calculate_balance_quality(y: np.ndarray) -> float:
    """Calculate the normalised Shannon entropy of the class distribution."""
    y_flat = y.astype(int).flatten()
    if len(y_flat) == 0:
        return 0.0 

    class_counts = np.bincount(y_flat)
    class_counts = class_counts[class_counts > 0]
    num_classes = len(class_counts)
    
    if num_classes <= 1:
        return 0.0

    proportions = class_counts / len(y_flat)
    entropy = -np.sum(proportions * np.log2(proportions + 1e-12))
    max_entropy = np.log2(num_classes)
    
    return float(entropy / max_entropy)
