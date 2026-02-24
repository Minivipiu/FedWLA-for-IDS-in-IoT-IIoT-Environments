Federated Intrusion Detection Datasets 🗄️

This directory is intended to store the preprocessed datasets used for evaluating the FedWLA aggregation strategy across different machine learning models (e.g., XGBoost, Random Forest, LightGBM, CatBoost, Decision Tree).

By placing the datasets here at the parent directory level, multiple algorithm-specific repositories can share the same underlying data files without duplicating gigabytes of information.

📊 Evaluated Datasets

The FedWLA architecture was rigorously evaluated on four representative, publicly available datasets that cover a wide range of IoT and IIoT network traffic characteristics.

To fully replicate the experiments detailed in the manuscript, researchers should procure and preprocess the following datasets:

1. Edge-IIoTset

Environment: IoT / IIoT

Description: A highly realistic dataset featuring traffic from various devices and multiple attack types. It is specifically designed to reflect realistic edge scenarios, making it highly suitable for evaluating intrusion detection models in both centralised and federated contexts.

2. X-IIoTID

Environment: IIoT

Description: A connectivity-agnostic and device-agnostic dataset focused on Industrial IoT. It provides statistical traffic features, network information, and protocol fields, enabling versatile multiclass evaluation.

3. WUSTL-IIoT-2021

Environment: IIoT

Description: Based on real traffic captured in industrial environments using multiple protocols (Modbus, OPC UA, MQTT, etc.). It is labelled at the network flow level, comprising benign sessions and simulated attack sequences.

4. IoT-23

Environment: IoT

Description: Contains 23 scenarios of benign and malicious traffic from IoT devices. It heavily features volumetric botnet attacks (such as Mirai and Gafgyt), facilitating robustness analysis against repetitive mass-scale attacks.

⚙️ Preprocessing Requirements

Please note that the federated simulation code expects these datasets to be strictly preprocessed before execution. Ensure that your CSV files meet the following criteria:

Fully Numerical: All categorical features must be encoded (e.g., one-hot encoding).

No Identifiers: Timestamps, IP addresses, MAC addresses, and session IDs must be removed to prevent data leakage and memorisation.

Cleaned: Null values and duplicate records should be addressed.

Label Column: The dataset must contain a target column (typically named Attack) with integer values representing the specific classes.

Place your preprocessed .csv files directly in this folder and update the dataset-path configuration in the pyproject.toml file of the corresponding experiment repository.
