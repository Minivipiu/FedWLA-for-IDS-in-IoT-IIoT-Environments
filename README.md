Federated Weighted Local Adaptation (FedWLA) for Intrusion Detection Systems 🌸

This repository contains the official implementation of FedWLA (Federated Weighted Local Adaptation) applied to Intrusion Detection Systems (IDS) in IoT and IIoT environments.

This code is part of the research presented in the manuscript: "An Adaptive Weighted Aggregation Strategy for Federated Intrusion Detection Systems in the Internet of Things".

📌 Overview

Traditional Federated Learning (FL) aggregation strategies, such as FedAvg, often struggle in highly heterogeneous and non-IID (Independent and Identically Distributed) edge environments. FedWLA overcomes these limitations by dynamically modulating each client's contribution to the global model based on three key local metrics:

Data Size ($N_i$): The volume of local samples.

Data Quality ($Q_i$): The normalised Shannon entropy of the local class distribution.

Predictive Uncertainty ($U_i$): The average entropy of the local model's predictions.

This repository provides the complete simulation environment to perform federated hyperparameter optimisation (HPO) across five distinct tree-based machine learning algorithms using the Flower (flwr) framework.

📂 Repository Structure

The project is structured into independent Flower Apps for each machine learning algorithm. They all share a central data/ directory to avoid duplicating heavy dataset files.

FedWLA-for-IDS/
├── data/                       # Shared preprocessed datasets directory
│   └── README.md               # Instructions for dataset download and preprocessing
├── XGB/                        # XGBoost implementation
│   ├── pyproject.toml
│   └── xgb_fedwla/
├── LGBM/                       # LightGBM implementation
│   ├── pyproject.toml
│   └── lgbm_fedwla/
├── CB/                         # CatBoost implementation
│   ├── pyproject.toml
│   └── cb_fedwla/
├── DT/                         # Decision Tree implementation
│   ├── pyproject.toml
│   └── dt_fedwla/
├── RF/                         # Random Forest implementation
│   ├── pyproject.toml
│   └── rf_fedwla/
├── LICENSE
└── README.md                   # This file


⚙️ Installation & Setup

Clone the repository:

git clone [https://github.com/Minivipiu/FedWLA-for-IDS-in-IoT-IIoT-Environments.git](https://github.com/Minivipiu/FedWLA-for-IDS-in-IoT-IIoT-Environments.git)
cd FedWLA-for-IDS-in-IoT-IIoT-Environments


Create a virtual environment and install dependencies:
It is highly recommended to use a Python virtual environment. Install the required packages to run all five implementations:

pip install flwr[simulation]>=1.19.0 numpy pandas scikit-learn xgboost lightgbm catboost filelock


📊 Dataset Preparation

Before running any simulation, you must procure and preprocess the datasets. We evaluated FedWLA on Edge-IIoTset, X-IIoTID, WUSTL-IIoT-2021, and IoT-23.

Please navigate to the data/ directory and read the README.md file for strict instructions on downloading, cleaning, applying logarithmic transformations, standardising, and one-hot encoding the data.

🛠️ Configuring a Custom Dataset

The architecture is highly modular. If you wish to test FedWLA on a new or different dataset, you only need to modify two aspects within the specific algorithm's folder (e.g., inside XGB/):

Update pyproject.toml: Change the dataset path and the number of simulated clients.

[tool.flwr.app.config]
num-server-rounds = 10
dataset-path = "../data/your_custom_dataset_cleaned.csv"
imbalance-factor = 5
num-partitions = 15  # Adjust based on your edge simulation needs


Update task.py:
Change the label_col parameter in the dataset loading functions to match your dataset's target variable (e.g., from "Traffic" or "Attack" to your specific column name).

def create_noniid_partitions_weighted(
    df: pd.DataFrame, num_clients: int, label_col: str = "YourTargetColumn",
    imbalance_factor: int = 5
):
# ...


🚀 Running the Simulations

Each algorithm is packaged as a standalone Flower App. To run an experiment, simply navigate into the desired algorithm's directory and launch the simulation.

For instance, to run the XGBoost experiment:

cd XGB
RAY_DISABLE_METRICS_COLLECTION=1 flwr run .


Note: The RAY_DISABLE_METRICS_COLLECTION=1 environment variable prevents harmless but noisy warnings regarding Ray's internal metric agents, ensuring a clean terminal output for your experiment logs.

What happens during execution?

Data Partitioning: The task.py module safely loads and partitions the dataset across the simulated clients (using FileLock to prevent race conditions).

Federated HPO: Clients perform local bounded random searches to evaluate hyperparameter configurations.

Adaptive Aggregation: In every communication round, the server calculates $N_i$, $Q_i$, and $U_i$ to dynamically weigh each client's optimal parameters and generate a robust global model.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.