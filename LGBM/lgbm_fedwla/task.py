"""Tasks module for data partitioning, caching, and local metrics calculation."""

import math
import pickle
import tempfile
from pathlib import Path

from filelock import FileLock
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

PARTITION_SEED = 42
CACHE_VERSION = "lgbm_noniid_nodup_safe_local_split_v2"


def create_noniid_partitions_weighted(
    df: pd.DataFrame,
    num_clients: int,
    label_col: str = "Attack",
    imbalance_factor: int = 5,
):
    """Create strictly non-overlapping non-IID partitions."""
    rng = np.random.default_rng(PARTITION_SEED)
    y = df[label_col].values
    classes = np.unique(y)

    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.array(idx_by_class[c], copy=True)
        n_c = len(idx_c)
        if n_c == 0:
            continue

        rng.shuffle(idx_c)

        if n_c >= num_clients:
            # Guarantee at least one sample per client whenever the data allow it.
            counts = np.ones(num_clients, dtype=int)
            remaining = n_c - num_clients

            if remaining > 0:
                raw_weights = rng.integers(
                    low=1,
                    high=imbalance_factor + 1,
                    size=num_clients,
                ).astype(float)
                weights = raw_weights / raw_weights.sum()
                expected_extra = weights * remaining
                extra_counts = np.floor(expected_extra).astype(int)
                counts += extra_counts

                remainder = remaining - int(extra_counts.sum())
                if remainder > 0:
                    fractional = expected_extra - extra_counts
                    order = np.argsort(-fractional)
                    for idx_client in order[:remainder]:
                        counts[idx_client] += 1
        else:
            # Assign the available samples to a random subset of clients.
            counts = np.zeros(num_clients, dtype=int)
            selected_clients = rng.permutation(num_clients)[:n_c]
            counts[selected_clients] = 1

        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            if end > start:
                client_indices[k].extend(idx_c[start:end])
            start = end

        if start != n_c:
            raise RuntimeError(f"Partitioning error for class {c!r}: assigned {start} != total {n_c}.")

    # Strict disjointness audit.
    seen = set()
    for idxs in client_indices:
        idx_set = set(idxs)
        if seen.intersection(idx_set):
            raise RuntimeError("Sample leakage detected across clients during partitioning.")
        seen.update(idx_set)

    return [df.iloc[np.sort(np.asarray(idxs, dtype=int))].copy() for idxs in client_indices]



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



def _safe_local_train_test_split(
    partition: pd.DataFrame,
    label_col: str = "Attack",
    test_size: float = 0.2,
    random_state: int = PARTITION_SEED,
):
    """Split a local partition into train/test while preserving strict isolation."""
    if len(partition) < 2:
        raise ValueError("A local partition must contain at least two samples to create train/test splits.")

    y = partition[label_col].astype(int)
    counts = y.value_counts().sort_index()
    n_samples = len(partition)
    n_classes = len(counts)

    n_test = int(math.ceil(test_size * n_samples)) if isinstance(test_size, float) else int(test_size)
    n_test = min(max(1, n_test), n_samples - 1)
    n_train = n_samples - n_test

    can_stratify = (
        counts.min() >= 2
        and n_test >= n_classes
        and n_train >= n_classes
    )

    if can_stratify:
        return train_test_split(
            partition,
            test_size=n_test,
            random_state=random_state,
            stratify=y,
        )

    rng = np.random.default_rng(random_state)
    grouped_indices = {
        cls: rng.permutation(partition.index[y == cls].to_numpy())
        for cls in counts.index
    }

    # Keep at least one sample of each local class in train whenever possible.
    train_indices = []
    remaining_indices = []
    for cls, idxs in grouped_indices.items():
        if len(idxs) == 0:
            continue
        train_indices.append(int(idxs[0]))
        if len(idxs) > 1:
            remaining_indices.extend(int(i) for i in idxs[1:])

    train_indices = list(dict.fromkeys(train_indices))
    remaining_indices = np.array(remaining_indices, dtype=int)
    if remaining_indices.size > 0:
        rng.shuffle(remaining_indices)

    target_train_size = n_train
    additional_train_needed = max(0, target_train_size - len(train_indices))

    additional_train = remaining_indices[:additional_train_needed].tolist()
    test_indices = remaining_indices[additional_train_needed:].tolist()
    train_indices.extend(additional_train)

    # If the rare-class constraint consumed too many samples, test could become
    # empty. In that case, move one non-essential sample from train to test.
    if len(test_indices) == 0:
        train_counts = pd.Series(partition.loc[train_indices, label_col].astype(int)).value_counts()
        movable = [
            idx for idx in train_indices
            if train_counts[partition.loc[idx, label_col]] > 1
        ]
        if not movable:
            raise ValueError(
                "Unable to create a non-empty test split without removing the only "
                "training example of at least one local class."
            )
        moved_idx = movable[-1]
        train_indices.remove(moved_idx)
        test_indices.append(moved_idx)

    train = partition.loc[sorted(train_indices)].copy()
    test = partition.loc[sorted(test_indices)].copy()

    return train, test



def get_partition(partition_id: int, num_partitions: int, dataset_path: str, imbalance_factor: int):
    """Load and cache dataset securely using FileLock for parallel simulation."""
    cache_dir = Path(tempfile.gettempdir()) / "fedwla_cache"
    cache_dir.mkdir(exist_ok=True)

    dataset_name = Path(dataset_path).stem
    cache_prefix = cache_dir / (
        f"part_{CACHE_VERSION}_{dataset_name}_{num_partitions}_{imbalance_factor}_{PARTITION_SEED}"
    )
    lock_file = cache_dir / (
        f"data_prep_{CACHE_VERSION}_{dataset_name}_{num_partitions}_{imbalance_factor}_{PARTITION_SEED}.lock"
    )
    target_cache_file = Path(f"{cache_prefix}_{partition_id}.pkl")

    with FileLock(lock_file):
        if not target_cache_file.exists():
            data = pd.read_csv(dataset_path, low_memory=False)
            data = data.dropna()

            label_col = "Attack" if "Attack" in data.columns else "Traffic"

            partitions = create_noniid_partitions_weighted(
                df=data,
                num_clients=num_partitions,
                label_col=label_col,
                imbalance_factor=imbalance_factor,
            )

            for i, p in enumerate(partitions):
                with open(f"{cache_prefix}_{i}.pkl", "wb") as f:
                    pickle.dump(p, f)

    with open(target_cache_file, "rb") as f:
        partition = pickle.load(f)

    label_col = "Attack" if "Attack" in partition.columns else "Traffic"
    local_y = partition[label_col].astype(int).values
    local_num_examples = len(partition)
    local_data_quality = calculate_balance_quality(local_y)

    train, test = _safe_local_train_test_split(
        partition,
        label_col=label_col,
        test_size=0.2,
        random_state=PARTITION_SEED,
    )

    # Intra-client leakage audit.
    train_idx = set(train.index.tolist())
    test_idx = set(test.index.tolist())
    if train_idx.intersection(test_idx):
        raise RuntimeError("Sample leakage detected between local train and test splits.")

    X_train = train.drop(columns=[label_col], errors="ignore").values
    y_train = train[label_col].astype(int).values
    X_test = test.drop(columns=[label_col], errors="ignore").values
    y_test = test[label_col].astype(int).values

    return X_train, y_train, X_test, y_test, local_num_examples, local_data_quality



def calculate_uncertainty(model: lgb.LGBMClassifier, X: np.ndarray) -> float:
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
