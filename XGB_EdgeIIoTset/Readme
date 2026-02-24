Federated Weighted Local Adaptation (FedWLA) for XGBoost 🌸

This repository contains the official implementation of FedWLA (Federated Weighted Local Adaptation) applied to XGBoost for Intrusion Detection Systems (IDS) in IoT/IIoT environments.

This code is part of the research presented in the manuscript: "An Adaptive Weighted Aggregation Strategy for Federated Intrusion Detection Systems in the Internet of Things".

📌 Overview

Traditional Federated Learning (FL) aggregation strategies, such as FedAvg, often struggle in highly heterogeneous and non-IID (Independent and Identically Distributed) edge environments. FedWLA overcomes these limitations by dynamically modulating each client's contribution to the global model based on three key local metrics:

Data Size ($N_i$): The volume of local samples.

Data Quality ($Q_i$): The normalised Shannon entropy of the local class distribution.

Predictive Uncertainty ($U_i$): The average entropy of the local model's predictions.

This specific implementation applies FedWLA to the hyperparameter optimisation (HPO) of XGBoost tree-based classifiers across simulated Multi-access Edge Computing (MEC) nodes using the Flower (flwr) framework.

📂 Project Structure

The repository is structured as a modern Flower App (flwr run . compatible):

├── pyproject.toml              # Flower application configuration and dependencies
├── README.md                   # This file
└── xgb_fedwla/                 # Main package directory
    ├── __init__.py
    ├── client.py               # Client-side logic (Local Random Search & Metrics)
    ├── metrics.py              # Server-side evaluation metrics aggregation
    ├── server.py               # Server-side logic and global parameter initialisation
    ├── strategy.py             # Core FedWLA aggregation strategy implementation
    └── task.py                 # Non-IID data partitioning, caching (FileLock), and maths


⚙️ Installation & Setup

Clone the repository:

git clone [https://github.com/Minivipiu/FedWLA-for-IDS-in-IoT-IIoT-Environments.git](https://github.com/Minivipiu/FedWLA-for-IDS-in-IoT-IIoT-Environments.git)
cd FedWLA-for-IDS-in-IoT-IIoT-Environments


Create a virtual environment and install dependencies:
It is highly recommended to use a virtual environment. The dependencies are managed automatically through the pyproject.toml file when running the project, but you can manually install them:

pip install flwr[simulation]>=1.19.0 numpy pandas scikit-learn xgboost filelock


📊 Dataset Preparation

By default, the project expects the preprocessed dataset (EdgeIIoT_cleaned.csv) to be located in a data folder at the parent directory level of this repository. This approach allows multiple models (e.g., RF, LGBM) to share the same heavy dataset without duplication.

Your directory tree should look like this:

├── data/
│   └── EdgeIIoT_cleaned.csv    <-- Place your dataset here
└── FedWLA-for-IDS-in-IoT-IIoT-Environments/
    ├── pyproject.toml
    └── xgb_fedwla/


Note: If you wish to use a different dataset or path, simply update the dataset-path variable inside the [tool.flwr.app.config] section of the pyproject.toml file.

🚀 Running the Simulation

To launch the federated simulation, ensure you are in the directory containing the pyproject.toml file and run the following command:

RAY_DISABLE_METRICS_COLLECTION=1 flwr run .


Note: The RAY_DISABLE_METRICS_COLLECTION=1 environment variable prevents harmless but noisy warnings regarding Ray's internal metric agents, ensuring a clean terminal output for your experiment logs.

What happens during the execution?

Data Caching: The task.py module uses FileLock to safely load, preprocess, and partition the dataset across the simulated clients only once, saving it to a temporary cache directory (e.g., /tmp/fedwla_cache/).

Federated Rounds: The server coordinates 10 communication rounds by default.

Local HPO: In each round, clients receive the global hyperparameters, perform a bounded random search, evaluate their local metrics ($Q_i, U_i$), and return their optimal configuration.

Adaptive Aggregation: The server applies the FedWLA strategy to generate a robust global configuration for the next round.


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
